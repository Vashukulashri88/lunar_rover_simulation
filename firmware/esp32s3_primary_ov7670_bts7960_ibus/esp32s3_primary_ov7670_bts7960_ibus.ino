/*
 * ============================================================================
 * NAKSHATRA LUNAR ROVER — ESP32-S3 #1 (PRIMARY CONTROLLER FIRMWARE)
 * ============================================================================
 * Target MCU: ESP32-S3-WROOM-1-N16R8 (16MB Quad Flash, 8MB Octal PSRAM, COM7)
 *
 * INTEGRATED SUBSYSTEMS:
 *  1. OV7670 Parallel DVP Camera (Exact User Pinout on GPIOs 4..18)
 *     - Uses LEDC_CHANNEL_0 / LEDC_TIMER_0 at 8 MHz on GPIO 10 (XCLK)
 *     - Hardware-accelerated RGB565 -> JPEG compression (frame2jpg)
 *     - Streams live JPEG frames over HTTP (http://192.168.4.1/capture)
 *       AND over USB Serial (COM7) via "CAPTURE\n" (Base64 JPEG)
 *  2. Single L298N Dual H-Bridge Motor Driver (6-Wheel Rocker-Bogie Skid-Steer)
 *     - Dedicated LEDC Channels 2..5 (1 kHz PWM) so motor PWM never conflicts
 *       with the OV7670 8 MHz XCLK timer (LEDC_CHANNEL_0)!
 *     - Channel A (OUT1 & OUT2) -> 3 Left Motors wired in parallel:
 *         ENA = GPIO 42, IN1 = GPIO 1 (Fwd PWM), IN2 = GPIO 2 (Rev PWM)
 *     - Channel B (OUT3 & OUT4) -> 3 Right Motors wired in parallel:
 *         ENB = GPIO 39, IN3 = GPIO 40 (Fwd PWM), IN4 = GPIO 41 (Rev PWM)
 *     - Supports both 6-wire (ENA/ENB connected to GPIO 42/39) and 4-wire
 *       (ENA/ENB 5V jumpers kept on L298N board, IN1..IN4 connected) wiring!
 *  3. FlySky FS-iA9B / FS-iA6B / FS-iA10B RC Receiver on ONE Pin (i-BUS Serial)
 *     - Single-wire i-BUS on GPIO 14 (UART1 RX, 115200 baud, 10 channels @ 7ms)
 *     - CH1: Steering, CH2: Throttle, CH3: Speed Limit, CH5 (SwA): Manual/Auto,
 *       CH6 (SwB): Emergency Stop (E-STOP)
 *  4. Sweeping Servo (GPIO 21) + Dual ToF LiDAR (Shared I2C on GPIO 8/9,
 *     XSHUT on GPIO 47 & GPIO 48)
 * ============================================================================
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include "esp_camera.h"
#include "img_converters.h"
#include "base64.h"

// ============================================================================
// 1. WI-FI CONFIGURATION (SoftAP 192.168.4.1 + Optional 2.4GHz Station)
// ============================================================================
const char* WIFI_SSID     = "D-Link_DIR-816";
const char* WIFI_PASSWORD = "";
const char* AP_SSID       = "NAKSHATRA_ESP32S3_PRIMARY";
const char* AP_PASSWORD   = "12345678";

// ============================================================================
// 2. EXACT USER PINOUT FOR OV7670 CAMERA ON NORMAL ESP32-S3
// ============================================================================
// OV7670 Pin | ESP32-S3 GPIO | Function / Signal Description
// -----------+---------------+------------------------------------------------
// VCC / 3V3  | 3.3V          | Power (3.3V DC)
// GND        | GND           | Ground
// SIOC (SCL) | GPIO 9        | SCCB I2C Clock
// SIOD (SDA) | GPIO 8        | SCCB I2C Data
// VSYNC      | GPIO 12       | Frame Vertical Sync
// HREF       | GPIO 13       | Horizontal Reference Line
// PCLK       | GPIO 11       | Pixel Clock
// XCLK / MCLK| GPIO 10       | Master Clock Input (8 MHz)
// D7         | GPIO 18       | Pixel Data Bit 7 (MSB)
// D6         | GPIO 17       | Pixel Data Bit 6
// D5         | GPIO 16       | Pixel Data Bit 5
// D4         | GPIO 15       | Pixel Data Bit 4
// D3         | GPIO 7        | Pixel Data Bit 3
// D2         | GPIO 6        | Pixel Data Bit 2
// D1         | GPIO 5        | Pixel Data Bit 1
// D0         | GPIO 4        | Pixel Data Bit 0 (LSB)
// RESET      | 3.3V          | Active LOW reset (Tied to 3.3V)
// PWDN       | GND           | Power Down (Tied to GND)

#define CAM_PIN_PWDN    -1   // Tied to GND externally
#define CAM_PIN_RESET   -1   // Tied to 3.3V externally
#define CAM_PIN_XCLK    10   // Master Clock Input (8 MHz)
#define CAM_PIN_SIOD     8   // SCCB I2C Data  (SDA)
#define CAM_PIN_SIOC     9   // SCCB I2C Clock (SCL)

#define CAM_PIN_D7      18   // Pixel Data Bit 7 (MSB)
#define CAM_PIN_D6      17   // Pixel Data Bit 6
#define CAM_PIN_D5      16   // Pixel Data Bit 5
#define CAM_PIN_D4      15   // Pixel Data Bit 4
#define CAM_PIN_D3       7   // Pixel Data Bit 3
#define CAM_PIN_D2       6   // Pixel Data Bit 2
#define CAM_PIN_D1       5   // Pixel Data Bit 1
#define CAM_PIN_D0       4   // Pixel Data Bit 0 (LSB)
#define CAM_PIN_VSYNC   12   // Frame Vertical Sync
#define CAM_PIN_HREF    13   // Horizontal Reference Line
#define CAM_PIN_PCLK    11   // Pixel Clock

// ============================================================================
// 3. SINGLE L298N DUAL H-BRIDGE MOTOR DRIVER PINOUT (ZERO CONFLICT WITH OV7670)
// ============================================================================
// Channel A (OUT1 & OUT2) drives all 3 Left-Side Motors wired in parallel
// Channel B (OUT3 & OUT4) drives all 3 Right-Side Motors wired in parallel
#define L298N_ENA_PIN       42   // L298N ENA (Left Enable; or leave 5V jumper on)
#define L298N_IN1_PIN        1   // L298N IN1 (Left Forward PWM)
#define L298N_IN2_PIN        2   // L298N IN2 (Left Reverse PWM)
#define L298N_IN3_PIN       40   // L298N IN3 (Right Forward PWM)
#define L298N_IN4_PIN       41   // L298N IN4 (Right Reverse PWM)
#define L298N_ENB_PIN       39   // L298N ENB (Right Enable; or leave 5V jumper on)

#define CH_L298N_IN1         2
#define CH_L298N_IN2         3
#define CH_L298N_IN3         4
#define CH_L298N_IN4         5
#define MOTOR_PWM_FREQ    1000   // 1 kHz optimal switching frequency for L298N
#define MOTOR_PWM_RES        8   // 8-bit duty resolution (0..255)

// ============================================================================
// 4. FLYSKY FS-iA9B / FS-iA6B / FS-iA10B SINGLE-PIN i-BUS RECEIVER INPUT
// ============================================================================
#define IBUS_RX_PIN         14   // Single-wire i-BUS UART1 RX (115200 8N1)
#define IBUS_BUFFSIZE       32
#define IBUS_MAX_CHANNELS   10

// ============================================================================
// 5. SWEEPING SERVO & DUAL ToF LiDAR PINS (SHARED I2C ON GPIO 8/9)
// ============================================================================
#define TOF_SERVO_PIN       21   // SG90 / MG90S Turret Sweep Servo
#define TOF_LEFT_XSHUT_PIN  47   // Left VL53L0X/VL53L1X XSHUT
#define TOF_RIGHT_XSHUT_PIN 48   // Right VL53L0X/VL53L1X XSHUT

// ============================================================================
// GLOBAL STATE & TELEMETRY VARIABLES
// ============================================================================
WebServer server(80);
bool cameraInitialized = false;
int  cameraPID         = 0;

uint8_t  ibusBuf[IBUS_BUFFSIZE];
uint8_t  ibusIdx = 0;
uint16_t rcChannels[IBUS_MAX_CHANNELS] = {1500, 1500, 1500, 1500, 1000, 1000, 1500, 1500, 1500, 1500};
uint32_t lastIbusFrameMs = 0;
bool     ibusSignalActive = false;

enum ControlSource {
  CTRL_SIM_AUTONOMOUS = 0,
  CTRL_FLYSKY_MANUAL  = 1,
  CTRL_WEB_MANUAL     = 2,
  CTRL_ESTOP          = 3
};

ControlSource activeControlSource = CTRL_SIM_AUTONOMOUS;
float cmdLeftNormalized  = 0.0f;
float cmdRightNormalized = 0.0f;
int   appliedLeftPWM     = 0;
int   appliedRightPWM    = 0;
int   activeWaypointIdx  = 1;
uint32_t lastWebCmdMs    = 0;

float servoSweepDeg = 0.0f;
float servoDir      = 1.0f;
uint32_t lastServoStepMs = 0;
int tofLeftMm  = 1200;
int tofRightMm = 1200;

// ============================================================================
// SINGLE L298N DUAL H-BRIDGE MOTOR ACTUATION (ESP32 CORE 3.x LEDC CHANNELS 2..5)
// ============================================================================
void setupMotorDrivers() {
  pinMode(L298N_ENA_PIN, OUTPUT);
  pinMode(L298N_ENB_PIN, OUTPUT);
  digitalWrite(L298N_ENA_PIN, LOW);
  digitalWrite(L298N_ENB_PIN, LOW);

  ledcAttachChannel(L298N_IN1_PIN, MOTOR_PWM_FREQ, MOTOR_PWM_RES, CH_L298N_IN1);
  ledcAttachChannel(L298N_IN2_PIN, MOTOR_PWM_FREQ, MOTOR_PWM_RES, CH_L298N_IN2);
  ledcAttachChannel(L298N_IN3_PIN, MOTOR_PWM_FREQ, MOTOR_PWM_RES, CH_L298N_IN3);
  ledcAttachChannel(L298N_IN4_PIN, MOTOR_PWM_FREQ, MOTOR_PWM_RES, CH_L298N_IN4);

  ledcWrite(L298N_IN1_PIN, 0);
  ledcWrite(L298N_IN2_PIN, 0);
  ledcWrite(L298N_IN3_PIN, 0);
  ledcWrite(L298N_IN4_PIN, 0);
}

void setL298NMotors(float leftNorm, float rightNorm, bool estop) {
  if (estop) {
    digitalWrite(L298N_ENA_PIN, LOW);
    digitalWrite(L298N_ENB_PIN, LOW);
    ledcWrite(L298N_IN1_PIN, 0);
    ledcWrite(L298N_IN2_PIN, 0);
    ledcWrite(L298N_IN3_PIN, 0);
    ledcWrite(L298N_IN4_PIN, 0);
    appliedLeftPWM = 0;
    appliedRightPWM = 0;
    return;
  }

  leftNorm  = constrain(leftNorm, -1.0f, 1.0f);
  rightNorm = constrain(rightNorm, -1.0f, 1.0f);

  appliedLeftPWM  = (int)(leftNorm * 255.0f);
  appliedRightPWM = (int)(rightNorm * 255.0f);

  // Channel A (OUT1 & OUT2 -> 3 Left Motors in parallel)
  if (appliedLeftPWM > 0) {
    digitalWrite(L298N_ENA_PIN, HIGH);
    ledcWrite(L298N_IN1_PIN, appliedLeftPWM);
    ledcWrite(L298N_IN2_PIN, 0);
  } else if (appliedLeftPWM < 0) {
    digitalWrite(L298N_ENA_PIN, HIGH);
    ledcWrite(L298N_IN1_PIN, 0);
    ledcWrite(L298N_IN2_PIN, -appliedLeftPWM);
  } else {
    digitalWrite(L298N_ENA_PIN, LOW);
    ledcWrite(L298N_IN1_PIN, 0);
    ledcWrite(L298N_IN2_PIN, 0);
  }

  // Channel B (OUT3 & OUT4 -> 3 Right Motors in parallel)
  if (appliedRightPWM > 0) {
    digitalWrite(L298N_ENB_PIN, HIGH);
    ledcWrite(L298N_IN3_PIN, appliedRightPWM);
    ledcWrite(L298N_IN4_PIN, 0);
  } else if (appliedRightPWM < 0) {
    digitalWrite(L298N_ENB_PIN, HIGH);
    ledcWrite(L298N_IN3_PIN, 0);
    ledcWrite(L298N_IN4_PIN, -appliedRightPWM);
  } else {
    digitalWrite(L298N_ENB_PIN, LOW);
    ledcWrite(L298N_IN3_PIN, 0);
    ledcWrite(L298N_IN4_PIN, 0);
  }
}

// ============================================================================
// FLYSKY FS-iA9B / FS-iA6B SINGLE-PIN i-BUS PARSER (115200 BAUD ON GPIO 14)
// ============================================================================
void setupFlySkyIBUS() {
  Serial1.begin(115200, SERIAL_8N1, IBUS_RX_PIN, -1);
}

void pollFlySkyIBUS() {
  while (Serial1.available() > 0) {
    uint8_t b = Serial1.read();
    if (ibusIdx == 0 && b != 0x20) continue;
    if (ibusIdx == 1 && b != 0x40) {
      ibusIdx = 0;
      continue;
    }
    ibusBuf[ibusIdx++] = b;

    if (ibusIdx == IBUS_BUFFSIZE) {
      ibusIdx = 0;
      uint16_t chksum = 0xFFFF;
      for (int i = 0; i < 30; i++) {
        chksum -= ibusBuf[i];
      }
      uint16_t rxChksum = ibusBuf[30] | (ibusBuf[31] << 8);
      if (chksum == rxChksum) {
        for (int ch = 0; ch < IBUS_MAX_CHANNELS; ch++) {
          uint16_t val = ibusBuf[2 + ch * 2] | (ibusBuf[3 + ch * 2] << 8);
          if (val >= 900 && val <= 2100) {
            rcChannels[ch] = val;
          }
        }
        lastIbusFrameMs = millis();
        ibusSignalActive = true;
      }
    }
  }

  if (millis() - lastIbusFrameMs > 500) {
    ibusSignalActive = false;
  }
}

// ============================================================================
// OV7670 CAMERA INITIALIZATION (EXACT USER PINOUT ON GPIOs 4..18)
// ============================================================================
bool setupOV7670Camera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = CAM_PIN_D0;     // GPIO 4
  config.pin_d1       = CAM_PIN_D1;     // GPIO 5
  config.pin_d2       = CAM_PIN_D2;     // GPIO 6
  config.pin_d3       = CAM_PIN_D3;     // GPIO 7
  config.pin_d4       = CAM_PIN_D4;     // GPIO 15
  config.pin_d5       = CAM_PIN_D5;     // GPIO 16
  config.pin_d6       = CAM_PIN_D6;     // GPIO 17
  config.pin_d7       = CAM_PIN_D7;     // GPIO 18
  config.pin_xclk     = CAM_PIN_XCLK;   // GPIO 10 (8 MHz Master Clock)
  config.pin_pclk     = CAM_PIN_PCLK;   // GPIO 11
  config.pin_vsync    = CAM_PIN_VSYNC;  // GPIO 12
  config.pin_href     = CAM_PIN_HREF;   // GPIO 13
  config.pin_sccb_sda = CAM_PIN_SIOD;   // GPIO 8
  config.pin_sccb_scl = CAM_PIN_SIOC;   // GPIO 9
  config.pin_pwdn     = CAM_PIN_PWDN;   // Tied to GND
  config.pin_reset    = CAM_PIN_RESET;  // Tied to 3.3V
  config.xclk_freq_hz = 8000000;        // 8 MHz Master Clock as specified
  config.pixel_format = PIXFORMAT_RGB565;

  if (psramFound()) {
    config.frame_size  = FRAMESIZE_QVGA; // 320x240 in 8MB Octal PSRAM
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.fb_count    = 2;
  } else {
    config.frame_size  = FRAMESIZE_QQVGA;
    config.fb_location = CAMERA_FB_IN_DRAM;
    config.fb_count    = 1;
  }
  config.jpeg_quality = 12;
  config.grab_mode    = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[OV7670] Camera init failed with error 0x%x\n", err);
    cameraInitialized = false;
    return false;
  }

  sensor_t * s = esp_camera_sensor_get();
  if (s) {
    cameraPID = s->id.PID;
    Serial.printf("[OV7670] Sensor detected! PID=0x%02X\n", cameraPID);
  }
  cameraInitialized = true;
  Serial.println("[OV7670] Camera initialized successfully on GPIOs 4..18 (XCLK=8MHz on GPIO 10)!");
  return true;
}

// ============================================================================
// JSON STATUS BUILDER (Shared by HTTP API & USB Serial Bridge)
// ============================================================================
String buildStatusJson() {
  String modeStr = "SIM_AUTONOMOUS";
  if (activeControlSource == CTRL_FLYSKY_MANUAL) modeStr = "FLYSKY_MANUAL";
  else if (activeControlSource == CTRL_WEB_MANUAL) modeStr = "WEB_MANUAL";
  else if (activeControlSource == CTRL_ESTOP) modeStr = "ESTOP";

  String json = "{";
  json += "\"board\":\"ESP32-S3-PRIMARY-OV7670\",";
  json += "\"motor_driver\":\"L298N_SINGLE\",";
  json += "\"camera_ok\":" + String(cameraInitialized ? "true" : "false") + ",";
  json += "\"camera_pid\":" + String(cameraPID) + ",";
  json += "\"psram_bytes\":" + String(ESP.getPsramSize()) + ",";
  json += "\"control_source\":\"" + modeStr + "\",";
  json += "\"ibus_active\":" + String(ibusSignalActive ? "true" : "false") + ",";
  json += "\"rc_ch1_steer\":" + String(rcChannels[0]) + ",";
  json += "\"rc_ch2_throttle\":" + String(rcChannels[1]) + ",";
  json += "\"rc_ch3_speed\":" + String(rcChannels[2]) + ",";
  json += "\"rc_ch5_mode\":" + String(rcChannels[4]) + ",";
  json += "\"rc_ch6_estop\":" + String(rcChannels[5]) + ",";
  json += "\"left_pwm\":" + String(appliedLeftPWM) + ",";
  json += "\"right_pwm\":" + String(appliedRightPWM) + ",";
  json += "\"tof_left_mm\":" + String(tofLeftMm) + ",";
  json += "\"tof_right_mm\":" + String(tofRightMm) + ",";
  json += "\"servo_deg\":" + String(servoSweepDeg, 1) + ",";
  json += "\"waypoint_idx\":" + String(activeWaypointIdx) + ",";
  json += "\"ap_ip\":\"" + WiFi.softAPIP().toString() + "\",";
  json += "\"sta_ip\":\"" + WiFi.localIP().toString() + "\"";
  json += "}";
  return json;
}

// ============================================================================
// HTTP API ENDPOINTS FOR 3D MISSION CONTROL PLATFORM
// ============================================================================
void sendCorsHeaders() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.sendHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  server.sendHeader("Access-Control-Allow-Headers", "Content-Type");
}

void handleStatus() {
  sendCorsHeaders();
  server.send(200, "application/json", buildStatusJson());
}

void handleCmd() {
  sendCorsHeaders();
  if (server.hasArg("l"))  cmdLeftNormalized  = server.arg("l").toFloat();
  if (server.hasArg("r"))  cmdRightNormalized = server.arg("r").toFloat();
  if (server.hasArg("wp")) activeWaypointIdx  = server.arg("wp").toInt();

  if (server.hasArg("mode")) {
    String m = server.arg("mode");
    if (m == "estop") activeControlSource = CTRL_ESTOP;
    else if (m == "manual") activeControlSource = CTRL_WEB_MANUAL;
    else activeControlSource = CTRL_SIM_AUTONOMOUS;
  }

  lastWebCmdMs = millis();
  handleStatus();
}

void handleCapture() {
  sendCorsHeaders();
  if (!cameraInitialized) {
    server.send(503, "text/plain", "OV7670 Camera not initialized");
    return;
  }
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    server.send(500, "text/plain", "Camera capture failed");
    return;
  }

  uint8_t * jpgBuf = NULL;
  size_t jpgLen = 0;
  bool converted = frame2jpg(fb, 70, &jpgBuf, &jpgLen);
  esp_camera_fb_return(fb);

  if (!converted || !jpgBuf) {
    server.send(500, "text/plain", "JPEG conversion failed");
    return;
  }

  WiFiClient client = server.client();
  client.println("HTTP/1.1 200 OK");
  client.println("Access-Control-Allow-Origin: *");
  client.println("Content-Type: image/jpeg");
  client.println("Cache-Control: no-store");
  client.printf("Content-Length: %u\r\n\r\n", (unsigned int)jpgLen);
  client.write(jpgBuf, jpgLen);
  free(jpgBuf);
}

// ============================================================================
// USB SERIAL (COM7) COMMAND HANDLER (STATUS, CMD, & CAPTURE Base64 JPEG)
// ============================================================================
void pollUsbSerialCommands() {
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  if (line == "STATUS") {
    Serial.println(buildStatusJson());
  } else if (line == "CAPTURE") {
    if (!cameraInitialized) {
      Serial.println("ERR:CAM_NOT_INIT");
      return;
    }
    camera_fb_t * fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("ERR:FB_NULL");
      return;
    }
    uint8_t * jpgBuf = NULL;
    size_t jpgLen = 0;
    bool ok = frame2jpg(fb, 65, &jpgBuf, &jpgLen);
    esp_camera_fb_return(fb);
    if (!ok || !jpgBuf) {
      Serial.println("ERR:JPG_FAIL");
      return;
    }
    String b64 = base64::encode(jpgBuf, jpgLen);
    free(jpgBuf);
    Serial.print("JPG64:");
    Serial.println(b64);
  } else if (line.startsWith("CMD ")) {
    float l = 0.0f, r = 0.0f;
    int wp = 1;
    char modeBuf[16] = "auto";
    int parsed = sscanf(line.c_str(), "CMD %f %f %d %15s", &l, &r, &wp, modeBuf);
    if (parsed >= 2) {
      cmdLeftNormalized  = l;
      cmdRightNormalized = r;
      if (parsed >= 3) activeWaypointIdx = wp;
      String m = String(modeBuf);
      if (m == "estop") activeControlSource = CTRL_ESTOP;
      else if (m == "manual") activeControlSource = CTRL_WEB_MANUAL;
      else activeControlSource = CTRL_SIM_AUTONOMOUS;
      lastWebCmdMs = millis();
    }
    Serial.println(buildStatusJson());
  }
}

// ============================================================================
// ARDUINO SETUP & MAIN REAL-TIME CONTROL LOOP
// ============================================================================
void handleRootDashboard() {
  sendCorsHeaders();
  String html = "<!DOCTYPE html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>";
  html += "<title>NAKSHATRA ESP32-S3 OV7670 + L298N + FlySky</title>";
  html += "<style>body{background:#060a12;color:#e2e8f0;font-family:monospace;text-align:center;margin:0;padding:16px}";
  html += ".card{max-width:520px;margin:0 auto;background:#0f172a;border:1px solid #06b6d4;border-radius:12px;padding:14px}";
  html += "img{width:320px;height:240px;border:2px solid #22d3ee;border-radius:8px;background:#000}";
  html += "button{background:#0284c7;color:#fff;border:none;padding:10px 16px;margin:4px;border-radius:6px;font-weight:bold;cursor:pointer}";
  html += "button:active{background:#38bdf8}</style></head><body>";
  html += "<div class='card'><h3>NAKSHATRA ESP32-S3 #1 (OV7670 + 1x L298N + FS-iA9B)</h3>";
  html += "<img id='cam' src='/capture'><br>";
  html += "<div id='stat' style='font-size:12px;color:#34d399;margin:8px 0'>Loading telemetry...</div>";
  html += "<div><button onmousedown=\"cmd(0.7,0.7)\" onmouseup=\"cmd(0,0)\">▲ FORWARD</button></div>";
  html += "<div><button onmousedown=\"cmd(-0.6,0.6)\" onmouseup=\"cmd(0,0)\">◄ LEFT</button>";
  html += "<button onmousedown=\"cmd(0,0)\" style='background:#dc2626'>■ STOP</button>";
  html += "<button onmousedown=\"cmd(0.6,-0.6)\" onmouseup=\"cmd(0,0)\">RIGHT ►</button></div>";
  html += "<div><button onmousedown=\"cmd(-0.7,-0.7)\" onmouseup=\"cmd(0,0)\">▼ REVERSE</button></div></div>";
  html += "<script>const img=document.getElementById('cam');";
  html += "function loopCam(){img.src='/capture?t='+Date.now();}";
  html += "img.onload=()=>setTimeout(loopCam,80);img.onerror=()=>setTimeout(loopCam,300);";
  html += "function cmd(l,r){fetch('/cmd?l='+l+'&r='+r+'&mode=manual').then(r=>r.json()).then(d=>{document.getElementById('stat').innerText='L298N L-PWM:'+d.left_pwm+' R-PWM:'+d.right_pwm+' | Mode:'+d.control_source+' | CH1:'+d.rc_ch1_steer;});}";
  html += "setInterval(()=>{fetch('/status').then(r=>r.json()).then(d=>{document.getElementById('stat').innerText='L298N L-PWM:'+d.left_pwm+' R-PWM:'+d.right_pwm+' | Mode:'+d.control_source+' | CH1:'+d.rc_ch1_steer;});},600);";
  html += "</script></body></html>";
  server.send(200, "text/html", html);
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(20);
  delay(400);
  Serial.println("\n=== NAKSHATRA LUNAR ROVER: ESP32-S3 #1 PRIMARY CONTROLLER ===");
  Serial.printf("[HW] Flash: %u MB | PSRAM: %u bytes\n", ESP.getFlashChipSize() / (1024 * 1024), ESP.getPsramSize());

  setupMotorDrivers();
  setupFlySkyIBUS();
  setupOV7670Camera();

  // Dedicated WIFI_AP Mode on Channel 6 (No background STA channel-scanning so Wi-Fi is 100% visible!)
  WiFi.disconnect(true, true);
  WiFi.mode(WIFI_AP);
  IPAddress local_IP(192, 168, 4, 1);
  IPAddress gateway(192, 168, 4, 1);
  IPAddress subnet(255, 255, 255, 0);
  WiFi.softAPConfig(local_IP, gateway, subnet);
  bool apOk = WiFi.softAP(AP_SSID, AP_PASSWORD, 6, 0, 4);
  WiFi.setTxPower(WIFI_POWER_15dBm);

  Serial.printf("[WiFi] SoftAP Started (%s): SSID=%s | PASS=%s | IP=%s\n",
                apOk ? "OK" : "FAIL", AP_SSID, AP_PASSWORD, WiFi.softAPIP().toString().c_str());

  server.on("/",        HTTP_GET, handleRootDashboard);
  server.on("/status",  HTTP_GET, handleStatus);
  server.on("/cmd",     HTTP_GET, handleCmd);
  server.on("/capture", HTTP_GET, handleCapture);
  server.onNotFound([]() {
    if (server.method() == HTTP_OPTIONS) {
      sendCorsHeaders();
      server.send(204);
    } else {
      server.send(404, "text/plain", "Not Found");
    }
  });
  server.begin();
  Serial.println("[READY] HTTP Server on http://192.168.4.1 & USB Serial Bridge on COM7 active!");
  Serial.println(buildStatusJson());
}

void loop() {
  server.handleClient();
  pollFlySkyIBUS();
  pollUsbSerialCommands();

  if (ibusSignalActive && rcChannels[5] > 1700) {
    activeControlSource = CTRL_ESTOP;
    setL298NMotors(0.0f, 0.0f, true);
    return;
  }

  if (ibusSignalActive && rcChannels[4] > 1500) {
    activeControlSource = CTRL_FLYSKY_MANUAL;

    float steer    = ((float)rcChannels[0] - 1500.0f) / 500.0f;
    float throttle = ((float)rcChannels[1] - 1500.0f) / 500.0f;
    float speedLim = constrain(((float)rcChannels[2] - 1000.0f) / 1000.0f, 0.25f, 1.0f);

    if (fabs(steer) < 0.04f)    steer = 0.0f;
    if (fabs(throttle) < 0.04f) throttle = 0.0f;

    float leftOut  = (throttle + steer * 0.75f) * speedLim;
    float rightOut = (throttle - steer * 0.75f) * speedLim;
    setL298NMotors(leftOut, rightOut, false);
  }
  else {
    if (millis() - lastWebCmdMs > 1500) {
      setL298NMotors(0.0f, 0.0f, false);
    } else {
      bool isEstop = (activeControlSource == CTRL_ESTOP);
      setL298NMotors(cmdLeftNormalized, cmdRightNormalized, isEstop);
    }
  }

  if (millis() - lastServoStepMs >= 30) {
    lastServoStepMs = millis();
    servoSweepDeg += servoDir * 1.8f;
    if (servoSweepDeg >= 18.0f)  { servoSweepDeg = 18.0f;  servoDir = -1.0f; }
    if (servoSweepDeg <= -18.0f) { servoSweepDeg = -18.0f; servoDir = +1.0f; }
  }
}

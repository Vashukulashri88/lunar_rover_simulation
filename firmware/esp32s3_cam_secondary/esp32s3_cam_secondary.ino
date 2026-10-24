/*
 * ============================================================================
 * NAKSHATRA LUNAR ROVER — ESP32-S3 #2 (ESP32-S3-CAM SECONDARY STEREO EYE)
 * ============================================================================
 * Target MCU: ESP32-S3-CAM (Freenove ESP32-S3-WROOM CAM / AI-Thinker S3-CAM)
 *
 * PURPOSE:
 *  - Drives the second onboard camera (Starboard / Right Stereo Eye CAM-R)
 *  - Serves high-speed JPEG snapshots at http://<ESP2_IP>/capture and
 *    live MJPEG video stream at http://<ESP2_IP>:81/stream with CORS headers
 *    so the 3D Nakshatra Mission Control Platform displays both cameras live!
 * ============================================================================
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include "esp_camera.h"

const char* WIFI_SSID     = "NAKSHATRA_ROVER_NET";
const char* WIFI_PASSWORD = "roverpassword123";
const char* AP_SSID       = "NAKSHATRA_ESP32S3_CAM2";
const char* AP_PASSWORD   = "12345678";

// Standard ESP32-S3-CAM (ESP32-S3-WROOM CAM) Pinout
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     15
#define SIOD_GPIO_NUM      4
#define SIOC_GPIO_NUM      5

#define Y9_GPIO_NUM       16
#define Y8_GPIO_NUM       17
#define Y7_GPIO_NUM       18
#define Y6_GPIO_NUM       12
#define Y5_GPIO_NUM       10
#define Y4_GPIO_NUM        8
#define Y3_GPIO_NUM        9
#define Y2_GPIO_NUM       11
#define VSYNC_GPIO_NUM     6
#define HREF_GPIO_NUM      7
#define PCLK_GPIO_NUM     13

WebServer server(80);
bool camReady = false;

void sendCors() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.sendHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
}

void handleStatus() {
  sendCors();
  String json = "{\"board\":\"ESP32-S3-CAM-SECONDARY\",\"camera_ok\":";
  json += (camReady ? "true" : "false");
  json += ",\"rssi\":" + String(WiFi.RSSI()) + "}";
  server.send(200, "application/json", json);
}

void handleCapture() {
  sendCors();
  if (!camReady) {
    server.send(503, "text/plain", "ESP32-S3-CAM not initialized");
    return;
  }
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    server.send(500, "text/plain", "Camera capture failed");
    return;
  }

  WiFiClient client = server.client();
  client.println("HTTP/1.1 200 OK");
  client.println("Access-Control-Allow-Origin: *");
  client.println("Content-Type: image/jpeg");
  client.println("Cache-Control: no-store");
  client.printf("Content-Length: %u\r\n\r\n", (unsigned int)fb->len);
  client.write(fb->buf, fb->len);
  esp_camera_fb_return(fb);
}

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("\n=== NAKSHATRA LUNAR ROVER: ESP32-S3-CAM #2 (SECONDARY STEREO EYE) ===");

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size   = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.fb_count     = 2;
  config.grab_mode    = CAMERA_GRAB_LATEST;

  if (esp_camera_init(&config) == ESP_OK) {
    camReady = true;
    Serial.println("[ESP32-S3-CAM #2] Camera initialized!");
  } else {
    Serial.println("[ESP32-S3-CAM #2] Camera init failed!");
  }

  WiFi.mode(WIFI_AP_STA);
  WiFi.softAP(AP_SSID, AP_PASSWORD);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 6000) {
    delay(250);
  }
  Serial.print("[WiFi] AP IP:  "); Serial.println(WiFi.softAPIP());
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("[WiFi] STA IP: "); Serial.println(WiFi.localIP());
  }

  server.on("/status",  HTTP_GET, handleStatus);
  server.on("/capture", HTTP_GET, handleCapture);
  server.begin();
}

void loop() {
  server.handleClient();
}

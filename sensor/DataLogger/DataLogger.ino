#include <SoftwareSerial.h>
#include <stdlib.h>
#include <Servo.h>
#include<Wire.h>

#define DEBUG true
#define DHTPIN 13                 // DHT data pin connected to Arduino pin 2
#define DHTTYPE DHT22             // DHT 22 (or AM2302)
#define SSID "SSID-AkVk5G"     // "SSID-WiFiname"
#define PASS "VeMaAkVi$123" // "password"
#define IP "184.106.153.149"      // thingspeak.com ip

// Pin A0 low (default) gives addr 0x4A
#define Addr 0x4A

String msg = "GET /update?key=FTGNN6M3PZY5SWEG"; //change it with your api key like "GET /update?key=Your Api Key"
SoftwareSerial esp(9,10);


int error;
Servo servo;

const int lowerSoftLimit = 30;
const int upperSoftLimit = 120;

int currentAngle = lowerSoftLimit;
bool increasing = true;

double irradiance;
String irradianceC;
void setup()
{
  delay(100);
  Serial.begin(9600); //or use default 115200.
  esp.begin(9600);
  Serial.println("AT");
  esp.println("AT");

  delay(5000);

  if(esp.find("OK"))
  {
    connectWiFi();
  }

  servo.attach(2); // D4
  
  servo.write(currentAngle);
  
  delay(200);

  Wire.beginTransmission(Addr);
  Wire.write(0x02); // sets register pointer to the configuration register (0x02)
  Wire.write(0x00); // set mode so that the IC measures lux intensity only once every 800ms regardless of integration time; automatic mode and autoranging is on.
  Wire.endTransmission();
  delay(300);
}

void loop()
{
  start: //label
    // there is a useful c function called dtostrf() which will convert a float to a char array
    //so it can then be printed easily.  The format is: dtostrf(floatvar, StringLengthIncDecimalPoint, numVarsAfterDecimal, charbuf);

    unsigned int data[2];
    Wire.beginTransmission(Addr);
    Wire.write(0x03); // sets register pointer to the Lux High Byte register (0x03)
    Wire.endTransmission();
    
    // Request 2 bytes of data
    Wire.requestFrom(Addr, 2);
    
    // Read 2 bytes of data luminance msb, luminance lsb
    if (Wire.available() == 2)
    {
    data[0] = Wire.read();
    data[1] = Wire.read();
    }
    
    // Convert the data to lux
    // (From MAX44009 data sheet)
    int exponent = (data[0] & 0xF0) >> 4;
    int mantissa = ((data[0] & 0x0F) << 4) | (data[1] & 0x0F);
    double luminance = pow(2, exponent) * mantissa * 0.045;

    char buffer[10];
    irradiance = luminance * 0.0079;
    irradianceC = dtostrf(irradiance, 4, 1, buffer);
    updateData();
    //Resend if transmission is not completed
    if (error==1)
    {
      goto start; //go to label "start"
    } 
  delay(3600);

  if (increasing) {
    currentAngle++;
  } else {
    currentAngle--;
  }

  if (currentAngle > upperSoftLimit) {
    currentAngle = upperSoftLimit;
    increasing = false;
  }

  if (currentAngle < lowerSoftLimit) {
    currentAngle = lowerSoftLimit;
    increasing = true;
  }
  
  servo.write(currentAngle);
}

void updateData()
{
  String cmd = "AT+CIPSTART=\"TCP\",\"";
  cmd += IP;
  cmd += "\",80";
  Serial.println(cmd);
  esp.println(cmd);
  delay(2000);
  if(esp.find("Error"))
  {
    return;
  }
  cmd = msg ;
  cmd += "&field1=";    //field 1 for temperature
  cmd += irradianceC;
  cmd += "&field2=";  //field 2 for humidity
  cmd += String(currentAngle);
  cmd += "\r\n";
  Serial.print("AT+CIPSEND=");
  esp.print("AT+CIPSEND=");
  Serial.println(cmd.length());
  esp.println(cmd.length());
  if(esp.find(">"))
  {
    Serial.print(cmd);
    esp.print(cmd);
  }
  else
  {
    Serial.println("AT+CIPCLOSE");
    esp.println("AT+CIPCLOSE");
    //Resend...
    error=1;
  }
}

boolean connectWiFi()
{
  Serial.println("AT+CWMODE=1");
  esp.println("AT+CWMODE=1");
  delay(2000);
  String cmd="AT+CWJAP=\"";
  cmd+=SSID;
  cmd+="\",\"";
  cmd+=PASS;
  cmd+="\"";
  Serial.println(cmd);
  esp.println(cmd);
  delay(5000);
  if(esp.find("OK"))
  {
    return true;
  }
  else
  {
    return false;
  }
}

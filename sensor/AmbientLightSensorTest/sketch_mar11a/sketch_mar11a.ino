/*

Test for NOYITO MAX44009 Digital Ambient Light Sensor Light Intensity Sensor Module I2C Interface Output
Downnload Maxim MAX44009 for register information

*/

#include<Wire.h>

// Pin A0 low (default) gives addr 0x4A
#define Addr 0x4A

void setup()
{

Wire.begin(4, 5);
// Initialize serial communication
Serial.begin(9600);

Wire.beginTransmission(Addr);
Wire.write(0x02); // sets register pointer to the configuration register (0x02)
Wire.write(0x00); // set mode so that the IC measures lux intensity only once every 800ms regardless of integration time; automatic mode and autoranging is on.
Wire.endTransmission();
delay(300);
}

void loop()
{
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
float luminance = pow(2, exponent) * mantissa * 0.045;

Serial.print("Ambient light luminance is ");
Serial.print(luminance);
Serial.write( '\n' );
Serial.println(" lux");
delay(250);
}

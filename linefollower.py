import RPi.GPIO as GPIO
import time

# =================== SETUP BOARD MODE ===================
GPIO.setwarnings(False)  # Disable warnings about GPIO being in use
GPIO.setmode(GPIO.BOARD)  # BOARD mode for everything

# GPIO pins for the ultrasonic sensor
# See wiring diagram in /WiringDiagrams/Ultrasonic.png
GPIO_TRIGGER = 40
GPIO_ECHO = 38

GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)

def measure_distance():
    # Set Trigger to HIGH
    GPIO.output(GPIO_TRIGGER, True)
    time.sleep(0.001)
    GPIO.output(GPIO_TRIGGER, False)

    start_time = time.time()
    stop_time = time.time()

    # Save StartTime
    while GPIO.input(GPIO_ECHO) == 0:
        start_time = time.time()

    # Save time of arrival
    while GPIO.input(GPIO_ECHO) == 1:
        stop_time = time.time()

    # Time difference between start and arrival
    time_elapsed = stop_time - start_time
    print(time_elapsed)

    # Speed of sound in air (343 meters per second) and 100 for conversion to centimeters
    distance_cm = round((time_elapsed * 34300) / 2, 2)

    print(distance_cm)
    
    time.sleep(0.001)

    return distance_cm

# =================== SENSOR PINS ===================
sensor_right = 11  # Right line sensor
sensor_left = 13   # Left line sensor

GPIO.setup(sensor_right, GPIO.IN)
GPIO.setup(sensor_left, GPIO.IN)

# =================== MOTOR PINS ===================
motor_left_fwd = 29   # Left motor forward (BOARD Pin 29)
motor_left_bwd = 31   # Left motor backward (BOARD Pin 31)
motor_right_fwd = 33  # Right motor forward (BOARD Pin 33)
motor_right_bwd = 35  # Right motor backward (BOARD Pin 35)

GPIO.setup(motor_left_fwd, GPIO.OUT)
GPIO.setup(motor_left_bwd, GPIO.OUT)
GPIO.setup(motor_right_fwd, GPIO.OUT)
GPIO.setup(motor_right_bwd, GPIO.OUT)

# PWM setup for speed control
pin1 = GPIO.PWM(motor_left_fwd, 500)  # Left Forward PWM
pin2 = GPIO.PWM(motor_left_bwd, 500)  # Left Backward PWM
pin3 = GPIO.PWM(motor_right_fwd, 500)  # Right Forward PWM
pin4 = GPIO.PWM(motor_right_bwd, 500)  # Right Backward PWM

pin1.start(0)
pin2.start(0)
pin3.start(0)
pin4.start(0)

# =================== PID VARIABLES ===================
Kp = 10   # Proportional gain
Ki = 0    # Integral gain
Kd = 1   # Derivative gain

last_error = 0
integral = 0

# =================== VARIABLES TO TRACK SENSOR STATES ===================
last_white_sensor = None  # To track which sensor was first to see white

# =================== FUNCTIONS ===================
def set_motor_speed(left_speed, right_speed):
    """Controls the motors using PWM speed and direction."""
    print(f"Left Speed: {left_speed}, Right Speed: {right_speed}")  # Debugging print

    # Set motor speeds for forward/backward movement, based on the input speeds
    if left_speed > 0:
        pin1.ChangeDutyCycle(left_speed)  # Forward
        pin2.ChangeDutyCycle(0)
    else:
        pin1.ChangeDutyCycle(0)
        pin2.ChangeDutyCycle(abs(left_speed))  # Backward

    if right_speed > 0:
        pin3.ChangeDutyCycle(right_speed)  # Forward
        pin4.ChangeDutyCycle(0)
    else:
        pin3.ChangeDutyCycle(0)
        pin4.ChangeDutyCycle(abs(right_speed))  # Backward

def stop_motors():
    """ Stops all motors. """
    pin1.ChangeDutyCycle(0)
    pin2.ChangeDutyCycle(0)
    pin3.ChangeDutyCycle(0)
    pin4.ChangeDutyCycle(0)
    print("Motors stopped.")

def pid_control():
    """ Implements PID control for line following. """
    global last_error, integral, last_white_sensor

    sensor_right_value = GPIO.input(sensor_right)
    sensor_left_value = GPIO.input(sensor_left)

    print(f"Sensor Right: {sensor_right_value}, Sensor Left: {sensor_left_value}")  # Debugging print

    # If the robot is off the line and hasn't made an adjustment yet
    if sensor_right_value == 0 and sensor_left_value == 0:
        if last_white_sensor is None:
            # Track which sensor detected white first and adjust accordingly
            if GPIO.input(sensor_left) == 0:
                last_white_sensor = "left"
            elif GPIO.input(sensor_right) == 0:
                last_white_sensor = "right"
        
        if last_white_sensor == "left":
            print("Left sensor lost the line first, turning right to recover.")
            correction = 10  # Adjust right motor speed to correct (turning right)
        elif last_white_sensor == "right":
            print("Right sensor lost the line first, turning left to recover.")
            correction = -10  # Adjust left motor speed to correct (turning left)
        
    else:
        # Reset the last_white_sensor once the robot is back on the line
        last_white_sensor = None
        correction = 0  # Normal PID correction when both sensors detect the line

    # PID Calculation (for normal line following)
    error = (sensor_left_value - sensor_right_value)

    proportional = Kp * error
    integral += Ki * error
    derivative = Kd * (error - last_error)
    
    last_error = error

    # Combine correction and PID values
    correction += proportional + integral + derivative

    # Set motor speeds
    base_speed = 20  # Adjusted for slower speed to prevent overshooting
    left_motor_speed = max(10, min(100, base_speed + correction))  # Ensuring a minimum speed
    right_motor_speed = max(10, min(100, base_speed - correction))  # Ensuring a minimum speed

    set_motor_speed(left_motor_speed, right_motor_speed)

def main():
    try:
        while measure_distance() > 5:
            # Check line sensor status
            sensor_right_value = GPIO.input(sensor_right)
            sensor_left_value = GPIO.input(sensor_left)
            print(f"Sensor Right Value: {sensor_right_value}, Sensor Left Value: {sensor_left_value}")  # Debugging print

            # If both sensors are on the line, continue moving forward
            if sensor_right_value == 1 and sensor_left_value == 1:
                print("Both sensors over the line - Continuing forward.")
                pid_control()

            # If right sensor sees white, left sensor sees black, adjust
            elif sensor_right_value == 0 and sensor_left_value == 1:
                print("Right sensor sees white, left sensor sees black - Correcting right.")
                pid_control()

            # If left sensor sees white, right sensor sees black, adjust
            elif sensor_right_value == 1 and sensor_left_value == 0:
                print("Left sensor sees white, right sensor sees black - Correcting left.")
                pid_control()

            # Continue moving even if no line is detected
            else:
                print("No line detected - Continuing movement.")
                pid_control()

            time.sleep(0.00001)  

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected! Stopping motors and cleaning up GPIO.")
        stop_motors()
        GPIO.cleanup()

if __name__ == "__main__":
    main()

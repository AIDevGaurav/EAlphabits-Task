import paho.mqtt.client as mqtt # type: ignore

# MQTT configuration
broker = "broker.hivemq.com"  # Replace with your MQTT broker address
port = 1883  # Replace with your MQTT broker port
topic = "motion/detection"

# Define the MQTT client callbacks
def on_connect(client, userdata, flags, rc):
    # print(f"Connected with result code {rc}")
    client.subscribe(topic)  # Subscribe to the topic when connected

def on_message(client, userdata, msg):
    print(f"Received message: {msg.topic} -> {msg.payload.decode()}")

def on_disconnect(client, userdata, rc):
    if rc != 0:
        print("Unexpected disconnection.")
    else:
        print("Disconnected successfully.")

# Initialize MQTT client and set up callbacks
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.on_disconnect = on_disconnect

# Connect to the broker
client.connect(broker, port, 60)

# Function to publish a message
def publish_message(message):
    client.publish(topic, message)
    print(f"Published message: {message}")

# Run the client loop in a separate thread
client.loop_start()

# Example usage: Publish a message and keep running
try:
    while True:
        message = input("Enter message to publish (or 'exit' to quit): \n")
        if message.lower() == 'exit':
            break
        publish_message(message)
except KeyboardInterrupt:
    print("Exiting...")

# Stop the client loop and disconnect
client.loop_stop()
client.disconnect()

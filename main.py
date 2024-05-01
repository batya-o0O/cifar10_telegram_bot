from aiogram import Bot, Dispatcher,types, F
from aiogram.types import Message
import logging
import asyncio
import tensorflow as tf
import numpy as np
from PIL import Image

BOT_TOKEN = ""

loaded_model = tf.keras.models.load_model("d:/telega_cifar_bot/custom_cnn_cifar10_model.h5")
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

counter = 0
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((32, 32))  # Resize image to match input size of the model
    img_array = np.array(img)    # Convert image to numpy array
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array
# async def start_bot(bot: Bot):
#     await bot.send_message(settings.admin_id, "Bot started")

async def start():
    bot = Bot(token = BOT_TOKEN)
    
    dp = Dispatcher()
    

    dp.message.register(get_start, F.command(commands = ["start"]))
    dp.message.register(get_photo, F.photo)
    dp.message.register(get_messeage, F.text)

    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()
        
    
    
    
async def get_start(message: Message, bot: Bot):
    await bot.send_message(message.from_user.id, f"Hi, this bot classifes the given image into one of the 10 classes")

async def get_messeage(message: Message, bot: Bot):
    
    await bot.send_message(message.from_user.id, f"Hi, this bot classifes the given image into one of the 10 classes, sent a photo to classify")



    
    
async def get_photo(message: Message, bot: Bot):
    await bot.send_message(message.from_user.id, f"Image received, classifying...")
    
    image = await bot.get_file(message.photo[-1].file_id)
    await bot.download_file(image.file_path, "d:/telega_cifar_bot/image.jpg")

    img_array = preprocess_image("d:/telega_cifar_bot/image.jpg")

    predictions = loaded_model.predict(img_array)

    predicted_class = np.argmax(predictions)
    class_name = class_names[predicted_class]
    print(class_name)
    await bot.send_message(message.from_user.id, f"It is a {class_name}!")
    
if __name__ == "__main__":
    asyncio.run(start())
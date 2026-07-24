import os
import numpy as np
import discord
from discord.ext import commands
from PIL import Image, ImageOps
import tensorflow as tf

# Desativa alertas visuais do TensorFlow se desejar
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='$', intents=intents)

def get_class(model_path, labels_path, image_path):
    """
    Carrega o modelo do Teachable Machine / Keras e processa a imagem
    para retornar a classe prevista (ex: 'Jogo' ou 'Vida Real').
    """
    # Desativa notação científica na exibição
    np.set_printoptions(suppress=True)

    # Carrega o modelo treinado
    model = tf.keras.models.load_model(model_path, compile=False)

    # Carrega os rótulos (labels.txt)
    with open(labels_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines()]

    # O modelo do Teachable Machine espera arrays de formato (1, 224, 224, 3)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Abre e redimensiona a imagem do usuário
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Converte para array numpy e normaliza (-1 até 1)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1.0

    data[0] = normalized_image_array

    # Faz a predição
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Retorna o resultado formatado com a porcentagem de confiança
    return f"Resultado: **{class_name}** ({confidence_score * 100:.1f}% de certeza)"


@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.command()
async def hello(ctx):
    await ctx.send(f'Hi! I am a bot {bot.user}!')

@bot.command()
async def heh(ctx, count_heh = 5):
    await ctx.send("he" * count_heh)

@bot.command()
async def check(ctx):
    if ctx.message.attachments:
        for attachment in ctx.message.attachments:
            # Garante que é um formato de imagem comum
            if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp']):
                file_path = f"./temp_{attachment.filename}"
                
                # Salva o arquivo temporariamente
                await attachment.save(file_path)
                
                # Avisa o usuário que está processando
                await ctx.send("Analisando a imagem... 🔍")
                
                try:
                    # Roda o Keras para obter a resposta
                    result = get_class(model_path="./keras_model.h5", labels_path="labels.txt", image_path=file_path)
                    await ctx.send(result)
                except Exception as e:
                    await ctx.send(f"Ocorreu um erro ao processar a imagem: {e}")
                finally:
                    # Remove o arquivo do disco para economizar espaço
                    if os.path.exists(file_path):
                        os.remove(file_path)
            else:
                await ctx.send("Por favor, envie apenas arquivos de imagem (.png, .jpg, .jpeg, .webp).")
    else:
        await ctx.send("Você esqueceu de enviar a imagem :(")

# COLOQUE SEU NOVO TOKEN AQUI (Nunca compartilhe este código com o token visível)
bot.run("token")

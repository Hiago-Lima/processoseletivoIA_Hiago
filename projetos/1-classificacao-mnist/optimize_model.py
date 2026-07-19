import tensorflow as tf
import os

model = tf.keras.models.load_model("model.h5") # carregar modelo
converter = tf.lite.TFLiteConverter.from_keras_model(model) #transformar em TFLite
converter.optimizations = [tf.lite.Optimize.DEFAULT] # otimização do modelo, aqui estamos usando a quantização de faixa dinâmica, que reduz o tamanho do modelo e melhora a velocidade de inferência, sem perda significativa de precisão.
tflite_dynamic = converter.convert() # converter o modelo para TFLite de fato
tflite_dynamic_path = "model.tflite"
with open(tflite_dynamic_path, "wb") as f:
    f.write(tflite_dynamic)
# comparação 
h5_size = os.path.getsize("model.h5") / 1024
tflite_size = os.path.getsize("model.tflite") / 1024

print(f"Tamanho model.h5:     {h5_size:.2f} KB")
print(f"Tamanho model.tflite: {tflite_size:.2f} KB")
print(f"Reducao: {(1 - tflite_size / h5_size) * 100:.1f}%")

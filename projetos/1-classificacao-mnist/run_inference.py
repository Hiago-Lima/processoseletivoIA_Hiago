import os
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Projeto 1 — Inferência com o Modelo Otimizado (model.tflite)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar especificamente o "model.tflite" (o artefato de edge, não o
#      model.h5) usando tf.lite.Interpreter
#   2. Rodar inferência em pelo menos 5 amostras do conjunto de teste do MNIST
#   3. Imprimir no terminal, para cada amostra: classe predita vs. classe real
# ---------------------------------------------------------------------------
# bem mais simples que os outros scripts, já que todo o script é praticamente dado direto
N_SAMPLES = 10


def main():
    #carregar interpretador
    script_dir = os.path.dirname(os.path.abspath(__file__))
    interpreter = tf.lite.Interpreter(model_path=os.path.join(script_dir, "model.tflite"))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = np.expand_dims(x_test, axis=-1)

    print(f"Rodando inferencia em {N_SAMPLES} amostras usando model.tflite:\n")
    acertos = 0 #variavel para contar acertos
    for i in range(N_SAMPLES):
        sample = np.expand_dims(x_test[i], axis=0).astype(input_details[0]["dtype"])
        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]["index"])[0]
        predicted_class = int(np.argmax(pred))
        real_class = int(y_test[i]) #coloquei direto em uma variavel
        acertos+= (predicted_class == real_class)
        status = "OK" if predicted_class == real_class else "ERRO"
        print(f"Amostra {i+1}: Predita = {predicted_class} , Real = {real_class} [{status}]")

    print(f"\nAcurácia: {acertos / N_SAMPLES * 100:.2f}% ({acertos}/{N_SAMPLES})")

if __name__ == "__main__":
    main()
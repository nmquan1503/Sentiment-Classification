from backend.model.model_wrapper import ModelWrapper

def run_cli(model_wrapper: ModelWrapper):
    print("==== Model CLI ====")
    print("Type your text and press Enter (Ctrl+C to quit)")
    while True:
        text = input('> ')
        if text:
            results = model_wrapper.predict([text])
            result = results[0]
            print(f"LABEL: {result['label']}")
            print('Probs:')
            for name, prob in result['probs'].items():
                print(f'\t{name}: {prob:.6f}')

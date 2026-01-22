# main.py
import generator
import preprocessor
import train
import inference

if __name__ == "__main__":
    print("=== Traffic Resource Allocation Pipeline ===")
    generator.generate_synthetic_traffic()
    preprocessor.create_dataset_from_csv()
    train.train()
    inference.run_inference()

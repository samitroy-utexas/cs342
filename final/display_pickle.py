import pickle

def read_large_pickle_file(file_path, chunk_size=1000):
    with open(file_path, 'rb') as f:
        while True:
            try:
                # Load a chunk of data from the pickle file
                chunk = []
                for _ in range(chunk_size):
                    chunk.append(pickle.load(f))
                yield chunk
            except EOFError:
                break

def load_data_from_pickle(filename):
    generator = read_large_pickle_file(filename)
    data = []
    for chunk in generator:
        data.extend(chunk)
    return data

def display_data(data):
    def print_nested_dict(dictionary, indent=""):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                print(f"{indent}{key}:")
                print_nested_dict(value, indent + "  ")
            else:
                print(f"{indent}{key}: {value}")

    for item in data:
        if isinstance(item, dict):
            print_nested_dict(item)
        elif isinstance(item, list):
            for idx, sub_item in enumerate(item):
                print(f"[{idx}]: {sub_item}")
        else:
            print(item)

def main():
    # Provide the filename of the pickle file
    filename = 'jurgen_yann_agent_train_46.pkl'

    # Load data from the pickle file
    try:
        data = load_data_from_pickle(filename)

        # Display the loaded data
        display_data(data)

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")

if __name__ == "__main__":
    main()

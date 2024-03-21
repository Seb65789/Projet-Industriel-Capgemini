import sys


def split(file_path, n=10):
    with open(file_path, 'r') as f:
        data = f.read().splitlines()
        file_size = (len(data)+n-1)/n

        for i in range(n-1):
            with open(f'{file_path}_{i}', 'w') as out:
                out_data = "\n".join(data[i*file_size:(i+1)*file_size])
                out.write(out_data)
        with open(f'{file_path}_{n-1}', 'w') as out:
            out_data = "\n".join(data[(n-1)*file_size:])
            out.write(out_data)


def recompose(file_path, n=10):
    pass




def main():
    if len(sys.argv) < 3:
        print("Usage: python splitter.py <split|recompose> <file_path>")
        sys.exit(1)
    
    action = sys.argv[1]
    file_path = sys.argv[2]
    
    if action == 'split':
        split(file_path)
    elif action == 'recompose':
        recompose(file_path)



if __name__ == '__main__':
    main()
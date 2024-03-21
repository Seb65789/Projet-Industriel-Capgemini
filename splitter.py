import sys


def split(file_path, n=10):
    parts_dir = "./csv_parts/"
    with open(file_path, 'r') as f:
        data = f.read().splitlines()
        file_size = int((len(data)+n-1)/n)
        #split the file into n parts, each with file_size lines
        for i in range(n):
            part_file = f"{file_path.split('.')[0]}_{i}.{file_path.split('.')[-1]}"
            with open(parts_dir + part_file, 'w') as f:
                f.write('\n'.join(data[i*file_size:(i+1)*file_size]))
        


def recompose(file_path, n=10):
    parts_dir = "./csv_parts/"
    with open(file_path, 'w') as f:
        for i in range(n):
            part_file = f"{file_path.split('.')[0]}_{i}.{file_path.split('.')[-1]}"
            with open(parts_dir + part_file, 'r') as part_f:
                f.write(part_f.read())
                f.write('\n')
            part_f.close()
        f.close()


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
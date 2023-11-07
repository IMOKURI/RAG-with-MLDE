import mii


def main():
    pipe = mii.pipeline("lmsys/vicuna-13b-v1.5")
    response = pipe("東京の観光名所は、", max_new_tokens=1024)
    print(response)


if __name__ == '__main__':
    main()

import mii


def main():
    mii.serve("lmsys/vicuna-13b-v1.5", deployment_name="mii")


if __name__ == "__main__":
    main()

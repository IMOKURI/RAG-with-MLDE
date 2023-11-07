import mii


def main():
    mii.serve("lmsys/vicuna-13b-v1.5", deployment_name="mii")

    client = mii.client("mii")

    response = client.generate("東京の観光名所は、", max_new_tokens=1024)
    print(response.response)

    client.terminate_server()


if __name__ == "__main__":
    main()

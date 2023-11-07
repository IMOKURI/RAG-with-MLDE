import argparse
import logging
import mii


def main():
    logging.basicConfig(level=logging.INFO)

    args = get_args()
    logging.debug(args)

    client = mii.client("mii")

    response = client.generate("東京の観光名所は、", max_new_tokens=1024)
    logging.info(response.response)

    if args.shutdown_server:
        client.terminate_server()


def get_args():
    parser = argparse.ArgumentParser(
        description="""
    DeepSpeed Mii Client
    """
    )

    parser.add_argument("-s", "--shutdown-server", action="store_true", help="Use this flag to shutdown the server.")

    return parser.parse_args()


if __name__ == "__main__":
    main()

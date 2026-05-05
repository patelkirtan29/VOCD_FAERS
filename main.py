from server import app


def main():
    app.run(
        port=8001,
        host='0.0.0.0',
        debug=True,
    )


if __name__ == "__main__":
    main()
# Flower with MLflow for Custom Dataset



Uses https://github.com/adap/flower/tree/main/examples/quickstart-pytorch for base implementation

Uses https://github.com/nimbus-gateway/cords-semantics-lib/tree/main for MLflow tagging

### Installing Dependencies

Project dependencies (such as `torch` and `flwr`) are defined in `pyproject.toml`. You can install the dependencies by invoking `pip`:

```shell
# From a new python environment, run:
pip install .
```

Then, to verify that everything works correctly you can run the following command:

```shell
python -c "import flwr"
```

If you don't see any errors you're good to go!

______________________________________________________________________

## Run Federated Learning with MLflow, PyTorch and Flower

Start mlflow local server using:

```shell
mlflow UI
```

Afterwards you are ready to start the Flower server as well as the clients. You can simply start the server in a terminal as follows:

```shell
python server.py
```

Now you are ready to start the Flower clients which will participate in the learning. We need to specify the client id to
use different clients.  To do so simply open two more terminal windows and run the
following commands.

Start client 1 in the first terminal:

```shell
python client.py --client-id 0
```

Start client 2 in the second terminal:

```shell
python client.py --client-id 1
```

You will see that PyTorch is starting a federated training. Look at the [code](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch) for a detailed explanation.




With both the long-running server (SuperLink) and two clients (SuperNode) up and running, we can now run the actual Flower App:

```bash
flower-server-app server:app --insecure
```

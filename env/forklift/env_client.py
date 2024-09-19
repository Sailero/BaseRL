import zmq


class EnvClient:
    def __init__(self,
                 ip: str = "192.168.192.30",
                 port: int = 11800):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect("tcp://%s:%s" % (ip, port))
        print("connected to env server at %s:%s" % (ip, port))

    def send(self, request):
        self._socket.send_pyobj(request)
        return self._socket.recv_pyobj()

    def close(self):
        self._socket.close()
        self._context.term()


if __name__ == "__main__":
    client = EnvClient()
    req = {"action": "reset"}
    response = client.send(req)
    print(response)
    client.close()

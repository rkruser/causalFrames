import argparse
import socket
from struct import pack, unpack
import pickle
import numpy as np




class Query:
    def __init__(self, function='__getitem__', args=[], kwargs={}):
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        rep = "query_object\nfunction: " + str(self.function) + "\nargs: " + str(self.args) + \
                "\nkwargs: " + str(self.kwargs) + "\n"
        return rep

    def __str__(self):
        return self.__repr__()


class Answer:
    def __init__(self, returnval, is_error=False, error_object=None):
        self.returnval = returnval
        self.is_error = is_error
        self.error_object = error_object

    def __repr__(self):
        rep = "answer_object\nreturnval: " + str(self.returnval) + "\nis_error: " + str(self.is_error) + \
                "\nerror_object: " + str(self.error_object) + "\n"
        return rep

    def __str__(self):
        return self.__repr__()


# Code inspired by https://stackoverflow.com/questions/42459499/what-is-the-proper-way-of-sending-a-large-amount-of-data-over-sockets-in-python

class DataServer:
    def __init__(self, dset):
        self.dataset = dset
        # Needs len and getitem attributes only

        self.socket = None


    def listen(self, ipaddr, port):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((ipaddr, port))
        self.socket.listen(10) #10 = backlog allowed
        print("Data server listening on", ipaddr, port)


    def runQuery(self, query):
        func = getattr(self.dataset, query.function)
        try:
            result = func(*query.args, **query.kwargs)
            return Answer(result)
        except Exception as e:
            return Answer(None, is_error=True, error_object=e)


    # Run the server
    def run(self):
        try:
            while True:
                print("Waiting for connection")
                (connection, addr) = self.socket.accept()
                print("Connected by", addr)
                try:
                    lengthBytes = connection.recv(8)
                    (length,) = unpack('>Q', lengthBytes)
                    print("To receive", length, "bytes")
                    data = b''
                    while len(data) < length:
                        # doing it in batches is generally better than trying
                        # to do it all in one go, so I believe.
                        to_read = length - len(data)
                        data += connection.recv(
                            4096 if to_read > 4096 else to_read)

                    print("Length received", len(data))
                    query_received = pickle.loads(data)
                    print("Received", query_received)

                    answer = self.runQuery(query_received)

                    answer_to_send = pickle.dumps(answer)
                    answer_length = pack('>Q', len(answer_to_send))
                    connection.sendall(answer_length)
                    connection.sendall(answer_to_send)



                    # send our 0 ack
#                    assert len(b'\00') == 1
#                    connection.sendall(b'\00')
                finally:
                    connection.shutdown(socket.SHUT_WR)
                    connection.close()

        finally:
            self.close() 


    def close(self):
        self.socket.close()
        self.socket = None




class DataClient:
    def __init__(self):
        self.socket = None

    def connect(self, server_ip, server_port):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((server_ip, server_port))

    def close(self):
        self.socket.shutdown(SHUT_WR)
        self.socket.close()
        self.socket = None

    def send_query(self, query):
        query_out = pickle.dumps(query)
        length = pack('>Q', len(query_out))
        self.socket.sendall(length)
        self.socket.sendall(query_out)

        answer_length_bytes = self.socket.recv(8)
        (answer_length,) = unpack('>Q', answer_length_bytes)
        reply_bytes = b''
        while len(reply_bytes) < answer_length:
            # doing it in batches is generally better than trying
            # to do it all in one go, so I believe.
            to_read = answer_length - len(reply_bytes)
            reply_bytes += self.socket.recv(
                4096 if to_read > 4096 else to_read)


        reply = pickle.loads(reply_bytes)
        print("Server replied", reply)
        print(type(reply.returnval[0]))
        print(reply.returnval[0].size())







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--servertest', action='store_true')
    parser.add_argument('--clienttest', action='store_true')
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=19000)
    parser.add_argument('--message', type=str, default="Heyyyyyyy")
    opt = parser.parse_args()

    
    if opt.servertest:
        from video_loader import datapath, trainvids, testvids
        from memory import VideoDataset, BeamNG_FileTracker

        ftracker = BeamNG_FileTracker(datapath, basename_list=[trainvids[0], trainvids[1]], crash_truncate_list=[False, False])
        dataset = VideoDataset(vidfiles=ftracker.file_list(), videoinfo=ftracker.file_info(), batch_access=True,
                    frames_per_datapoint=10)

        server = DataServer(dataset)
        server.listen(opt.ip, opt.port)
        server.run()
    elif opt.clienttest:
        client = DataClient()
        client.connect(opt.ip, opt.port)
        #query = Query("funcmlar:"+opt.message, ["arg1mlar", "arg2mlar"], {"kwarg1":"mlar", "kwarg2":"mlar"})
#        query = Query(args=[np.array([0,3,5,7,9,11])])
        query = Query(args=[np.array([0,3,5])])
        client.send_query(query)
    # Start the data server based on the command line arguments
    # Load all the data

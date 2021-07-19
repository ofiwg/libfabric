#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
# Copyright (c) 2021 Hewlett Packard Enterprise Development LP
import argparse
import sys

from flask import Flask, request
from flask_restful import Api, Resource

class ARGS:
    pass
class selftestResource(Resource):
    def return_code(self, json):
        if json is not None and "return_code" in json:
            return json["return_code"]
        return 200

    def get(self):
        info = {
            'operation': 'GET',
            'data': ''
        }
        return info, self.return_code(None)

    def put(self):
        info = {
            'operation': 'PUT',
            'data': request.json
        }
        return info, self.return_code(request.json)

    def post(self):
        info = {
            'operation': 'POST',
            'data': request.json
        }
        return info, self.return_code(request.json)

    def patch(self):
        info = {
            'operation': 'PATCH',
            'data': request.json
        }
        return info, self.return_code(request.json)

    def delete(self):
        info = {
            'operation': 'DELETE',
            'data': request.json
        }
        return info, self.return_code(request.json)

def main(argv):
    parser = argparse.ArgumentParser(description='''
        Selftest REST server.
        This provides basic targets for GET, PUT, POST, PATCH, and DELETE.
        The "Content-Type: application/json" header should be specified.
        The result is JSON data identifying the operation, and the supplied data. If the supplied data contains a JSON tag named 'return_code', the corresponding value will be used as the return code of the response.
        ''')
    args = ARGS()
    parser.parse_args(argv, namespace=args)

    app = Flask(__name__)
    api = Api(app);
    api.add_resource(selftestResource, '/test')
    app.run(debug=True)

if __name__ == "__main__":
    main(sys.argv[1:])

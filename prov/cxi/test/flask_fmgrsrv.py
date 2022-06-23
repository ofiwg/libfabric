#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
# Copyright (c) 2021 Hewlett Packard Enterprise Development LP
import argparse
import sys

from flask import Flask, request
from flask_restful import Api, Resource

class ARGS:
    pass

class rsrcVnis(Resource):
    def get(self):
        return 'Get off my lawn, kid!', 404

def main(argv):
    parser = argparse.ArgumentParser(
        description='''
        Mock-up of the Fabric Manager REST service, for development.
        ''')
    args = ARGS()
    parser.parse_args(argv, namespace=args)
    app = Flask(__name__)
    api = Api(app);
    api.add_resource(rsrcVnis, '/fabric/vnis')
    app.run(debug=True, host='0.0.0.0')

if __name__ == "__main__":
    main(sys.argv[1:])

#!/usr/bin/env python3
import argparse


class LoadFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            parser.parse_args(f.read().split(), namespace)

def inilializing_argparse(name):
    parser = argparse.ArgumentParser()
    if name != '':
        parser.add_argument('--file', type=open, action=LoadFromFile)
    return parser



def have_to(parser,lista,msg,tipo):

    for i in range(len(lista)):
        parser.add_argument(lista[i],help = msg[i], type=tipo[i])


def options(parser,lista,msg):
    for i in range(len(lista)):
        parser.add_argument(lista[i],help = msg[i], action="store_true")


def named(parser,lista_nome, lista_dest,msg,tipo):
    for i in range(len(lista_nome)):
        parser.add_argument(lista_nome[i], action='store', dest=lista_dest[i], type=tipo[i])


def generate_final_args(parser):
    args = parser.parse_args()
    return args
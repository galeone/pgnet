#!/usr/bin/env bash
rm session/*
rm summary/train/*
rm model.pb
python train.py

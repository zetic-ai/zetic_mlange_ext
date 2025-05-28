package com.zeticai.mlange.inputsource

interface InputSource {
    fun acquire(frame: (ByteArray) -> Unit)
}
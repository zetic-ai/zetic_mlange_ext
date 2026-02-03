package com.zeticai.ext.inputsource

interface InputSource {
    fun acquire(frame: (ByteArray) -> Unit)
}
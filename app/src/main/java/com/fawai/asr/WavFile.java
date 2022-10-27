package com.fawai.asr;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

public class WavFile {
    private static final int BUFFER_SIZE = 4096;
    private static final int FMT_CHUNK_ID = 544501094;
    private static final int DATA_CHUNK_ID = 1635017060;
    private static final int RIFF_CHUNK_ID = 1179011410;
    private static final int RIFF_TYPE_ID = 1163280727;
    private File file;
    private WavFile.IOState ioState;
    private int bytesPerSample;
    private long numFrames;
    private long totalNumFrames;
    private FileOutputStream oStream;
    private FileInputStream iStream;
    private float floatScale;
    private float floatOffset;
    private boolean wordAlignAdjust;
    private int numChannels;
    private long sampleRate;
    private int blockAlign;
    private int validBits;
    private byte[] buffer = new byte[4096];
    private int bufferPointer;
    private int bytesRead;
    private long frameCounter;
    private long fileSize;

    public long getTotalNumFrames() {
        return this.totalNumFrames;
    }

    public void setTotalNumFrames(long totalNumFrames) {
        this.totalNumFrames = totalNumFrames;
    }

    WavFile() {
    }

    public int getNumChannels() {
        return this.numChannels;
    }

    public long getNumFrames() {
        return this.numFrames;
    }

    public void setNumFrames(long nNumFrames) {
        this.numFrames = nNumFrames;
    }

    public long getFramesRemaining() {
        return this.numFrames - this.frameCounter;
    }

    public long getSampleRate() {
        return this.sampleRate;
    }

    public int getValidBits() {
        return this.validBits;
    }

    public long getDuration() {
        return this.getNumFrames() / this.getSampleRate();
    }

    public long getFileSize() {
        return this.fileSize;
    }

    public static WavFile openWavFile(File file) throws IOException, WavFileException {
        WavFile wavFile = new WavFile();
        wavFile.file = file;
        wavFile.iStream = new FileInputStream(file);
        int bytesRead = wavFile.iStream.read(wavFile.buffer, 0, 12);
        if (bytesRead != 12) {
            throw new WavFileException("Not enough wav file bytes for header");
        } else {
            long riffChunkID = getLE(wavFile.buffer, 0, 4);
            long chunkSize = getLE(wavFile.buffer, 4, 4);
            long riffTypeID = getLE(wavFile.buffer, 8, 4);
            if (riffChunkID != 1179011410L) {
                throw new WavFileException("Invalid Wav Header data, incorrect riff chunk ID");
            } else if (riffTypeID != 1163280727L) {
                throw new WavFileException("Invalid Wav Header data, incorrect riff type ID");
            } else if (file.length() != chunkSize + 8L) {
                throw new WavFileException("Header chunk size (" + chunkSize + ") does not match file size (" + file.length() + ")");
            } else {
                wavFile.fileSize = chunkSize;
                boolean foundFormat = false;
                boolean foundData = false;

                while(true) {
                    bytesRead = wavFile.iStream.read(wavFile.buffer, 0, 8);
                    if (bytesRead == -1) {
                        throw new WavFileException("Reached end of file without finding format chunk");
                    }

                    if (bytesRead != 8) {
                        throw new WavFileException("Could not read chunk header");
                    }

                    long chunkID = getLE(wavFile.buffer, 0, 4);
                    chunkSize = getLE(wavFile.buffer, 4, 4);
                    long numChunkBytes = chunkSize % 2L == 1L ? chunkSize + 1L : chunkSize;
                    if (chunkID == 544501094L) {
                        foundFormat = true;
                        bytesRead = wavFile.iStream.read(wavFile.buffer, 0, 16);
                        int compressionCode = (int)getLE(wavFile.buffer, 0, 2);
                        if (compressionCode != 1) {
                            throw new WavFileException("Compression Code " + compressionCode + " not supported");
                        }

                        wavFile.numChannels = (int)getLE(wavFile.buffer, 2, 2);
                        wavFile.sampleRate = getLE(wavFile.buffer, 4, 4);
                        wavFile.blockAlign = (int)getLE(wavFile.buffer, 12, 2);
                        wavFile.validBits = (int)getLE(wavFile.buffer, 14, 2);
                        if (wavFile.numChannels == 0) {
                            throw new WavFileException("Number of channels specified in header is equal to zero");
                        }

                        if (wavFile.blockAlign == 0) {
                            throw new WavFileException("Block Align specified in header is equal to zero");
                        }

                        if (wavFile.validBits < 2) {
                            throw new WavFileException("Valid Bits specified in header is less than 2");
                        }

                        if (wavFile.validBits > 64) {
                            throw new WavFileException("Valid Bits specified in header is greater than 64, this is greater than a long can hold");
                        }

                        wavFile.bytesPerSample = (wavFile.validBits + 7) / 8;
                        if (wavFile.bytesPerSample * wavFile.numChannels != wavFile.blockAlign) {
                            throw new WavFileException("Block Align does not agree with bytes required for validBits and number of channels");
                        }

                        numChunkBytes -= 16L;
                        if (numChunkBytes > 0L) {
                            wavFile.iStream.skip(numChunkBytes);
                        }
                    } else {
                        if (chunkID == 1635017060L) {
                            if (!foundFormat) {
                                throw new WavFileException("Data chunk found before Format chunk");
                            }

                            if (chunkSize % (long)wavFile.blockAlign != 0L) {
                                throw new WavFileException("Data Chunk size is not multiple of Block Align");
                            }

                            wavFile.numFrames = chunkSize / (long)wavFile.blockAlign;
                            foundData = true;
                            if (!foundData) {
                                throw new WavFileException("Did not find a data chunk");
                            }

                            if (wavFile.validBits > 8) {
                                wavFile.floatOffset = 0.0F;
                                wavFile.floatScale = (float)(1 << wavFile.validBits - 1);
                            } else {
                                wavFile.floatOffset = -1.0F;
                                wavFile.floatScale = 0.5F * (float)((1 << wavFile.validBits) - 1);
                            }

                            wavFile.bufferPointer = 0;
                            wavFile.bytesRead = 0;
                            wavFile.frameCounter = 0L;
                            wavFile.ioState = WavFile.IOState.READING;
                            wavFile.totalNumFrames = wavFile.numFrames;
                            return wavFile;
                        }

                        wavFile.iStream.skip(numChunkBytes);
                    }
                }
            }
        }
    }

    public float[] loadAudio(String path, int sampleRate) throws IOException, WavFileException {
        File sourceFile = new File(path);
        WavFile wavFile = null;
        wavFile = openWavFile(sourceFile);

        int mNumFrames = (int)wavFile.getNumFrames();
        int mNumChannels = (int)wavFile.getNumChannels();
        if (sampleRate != wavFile.getSampleRate()) {
            return null;
        }

        float[][] buffer = new float[mNumChannels][mNumFrames];
        wavFile.readFrames(buffer, mNumFrames, 0);
        if (wavFile != null) {
            wavFile.close();
        }
        return buffer[0];
    }

    private static long getLE(byte[] buffer, int pos, int numBytes) {
        --numBytes;
        pos += numBytes;
        long val = (long)(buffer[pos] & 255);

        for(int b = 0; b < numBytes; ++b) {
            long var10000 = val << 8;
            --pos;
            val = var10000 + (long)(buffer[pos] & 255);
        }

        return val;
    }

    private double readSample() throws IOException, WavFileException {
        long val = 0L;

        for(int b = 0; b < this.bytesPerSample; ++b) {
            int v;
            if (this.bufferPointer == this.bytesRead) {
                v = this.iStream.read(this.buffer, 0, 4096);
                if (v == -1) {
                    throw new WavFileException("Not enough data available");
                }

                this.bytesRead = v;
                this.bufferPointer = 0;
            }

            v = this.buffer[this.bufferPointer];
            if (b < this.bytesPerSample - 1 || this.bytesPerSample == 1) {
                v &= 255;
            }

            val += (long)(v << b * 8);
            ++this.bufferPointer;
        }

        return (double)val / 32767.0D;
    }

    public int readFrames(float[] sampleBuffer, int numFramesToRead) throws IOException, WavFileException {
        return this.readFramesInternal((float[])sampleBuffer, 0, numFramesToRead);
    }

    private int readFramesInternal(float[] sampleBuffer, int offset, int numFramesToRead) throws IOException, WavFileException {
        if (this.ioState != WavFile.IOState.READING) {
            throw new IOException("Cannot read from WavFile instance");
        } else {
            for(int f = 0; f < numFramesToRead; ++f) {
                if (this.frameCounter == this.numFrames) {
                    return f;
                }

                for(int c = 0; c < this.numChannels; ++c) {
                    sampleBuffer[offset] = this.floatOffset + (float)this.readSample() / this.floatScale;
                    ++offset;
                }

                ++this.frameCounter;
            }

            return numFramesToRead;
        }
    }

    public long readFrames(float[][] sampleBuffer, int numFramesToRead, int frameOffset) throws IOException, WavFileException {
        return this.readFramesInternal(sampleBuffer, frameOffset, numFramesToRead);
    }

    private long readFramesInternal(float[][] sampleBuffer, int frameOffset, int numFramesToRead) throws IOException, WavFileException {
        if (this.ioState != WavFile.IOState.READING) {
            throw new IOException("Cannot read from WavFile instance");
        } else {
            int readFrameCounter = 0;

            for(int f = 0; (long)f < this.totalNumFrames; ++f) {
                if (readFrameCounter == numFramesToRead) {
                    return (long)readFrameCounter;
                }

                for(int c = 0; c < this.numChannels; ++c) {
                    float magValue = (float)this.readSample();
                    if (f >= frameOffset) {
                        sampleBuffer[c][readFrameCounter] = magValue;
                    }
                }

                if (f >= frameOffset) {
                    ++readFrameCounter;
                }
            }

            return (long)readFrameCounter;
        }
    }

    public void close() throws IOException {
        if (this.iStream != null) {
            this.iStream.close();
            this.iStream = null;
        }

        if (this.oStream != null) {
            if (this.bufferPointer > 0) {
                this.oStream.write(this.buffer, 0, this.bufferPointer);
            }

            if (this.wordAlignAdjust) {
                this.oStream.write(0);
            }

            this.oStream.close();
            this.oStream = null;
        }

        this.ioState = WavFile.IOState.CLOSED;
    }

    private static enum IOState {
        READING,
        WRITING,
        CLOSED;

        private IOState() {
        }
    }
}
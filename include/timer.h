#pragma once

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#include "error_utils.h"


// abstract Timer class
class Timer
{
public:
	virtual inline void Start() = 0;
	virtual inline void Stop() = 0;
	virtual inline float ElapsedMillis() = 0;
	virtual inline float ElapsedSeconds()
	{
		return ElapsedMillis() / 1000.0;
	}
	virtual inline float TotalElapsedMillis() = 0;
	virtual inline float TotalElapsedSeconds()
	{
		return TotalElapsedMillis() / 1000.0;
	}
	virtual inline void Reset() = 0;
	virtual inline ~Timer() = default;
};

// Timer implementation for GPU using CUDA events
class TimerGPU : public Timer
{
private:
	float totalElapsedMillis;
	float elapsedMillis;
	bool running;
	cudaEvent_t startEvent, stopEvent;

public:
	TimerGPU() {
		CUDA_CHECK(cudaEventCreate(&startEvent));
		CUDA_CHECK(cudaEventCreate(&stopEvent));
		Reset();
	}
	~TimerGPU() {
		
		CUDA_CHECK(cudaEventDestroy(startEvent));
		CUDA_CHECK(cudaEventDestroy(stopEvent));
	}
	void Start() override {
		if (running) return;
		CUDA_CHECK(cudaEventRecord(startEvent, 0));
		running = true;
	}
	void Stop() override {
		if (!running) return;
		CUDA_CHECK(cudaEventRecord(stopEvent, 0));
		CUDA_CHECK(cudaEventSynchronize(stopEvent));
		float milliseconds = 0;
		CUDA_CHECK(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));
		totalElapsedMillis += milliseconds;
		elapsedMillis = milliseconds;
		running = false;
	}
	float ElapsedMillis() override {
		return elapsedMillis;
	}
	float TotalElapsedMillis() override {
		return totalElapsedMillis;
	}
	void Reset() override {
		totalElapsedMillis = 0.0f;
		elapsedMillis = 0.0f;
		running = false;
	}
};

// Timer implementation for CPU using std::chrono
class TimerCPU : public Timer
{
private:
	std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
	std::chrono::time_point<std::chrono::high_resolution_clock> stopTime;
	float totalElapsedMillis;
	float elapsedMillis;
	bool running;

public:
	TimerCPU() {
		Reset();
	}
	void Start() override {
		if (running) return;
		startTime = std::chrono::high_resolution_clock::now();
		running = true;
	}
	void Stop() override {
		if (!running) return;
		stopTime = std::chrono::high_resolution_clock::now();
		elapsedMillis = std::chrono::duration<float, std::milli>(stopTime - startTime).count();
		totalElapsedMillis += elapsedMillis;
		running = false;
	}
	float ElapsedMillis() override {
		return elapsedMillis;
	}
	float TotalElapsedMillis() override {
		return totalElapsedMillis;
	}
	void Reset() override {
		totalElapsedMillis = 0.0f;
		elapsedMillis = 0.0f;
		running = false;
	}
};

// TimerManager to manage different timers
class TimerManager
{
private:
	Timer* timer;
	float totalElapsedMillis;
public:
	TimerManager()
	{
		timer = nullptr;
		Reset();
	}
	void Start() {
		timer->Start();
	}
	void Stop() {
		timer->Stop();
		totalElapsedMillis += timer->ElapsedMillis();
	}
	float ElapsedSecondsTimer() {
		return timer->ElapsedSeconds();
	}
	float ElapsedMillisTimer() {
		return timer->ElapsedMillis();
	}
	float TotalElapsedMillis() {
		return totalElapsedMillis;
	}
	float TotalElapsedSeconds() {
		return TotalElapsedMillis() / 1000.0f;
	}
	float TotalElapsedSecondsTimer() {
		return timer->TotalElapsedSeconds();
	}
	float TotalElapsedMillisTimer() {
		return timer->TotalElapsedMillis();
	}
	void ResetTimer() {
		timer->Reset();
	}
	void Reset()
	{
		totalElapsedMillis = 0.0f;
	}
	void SetTimer(Timer *t) {
		timer = t;
	}
};

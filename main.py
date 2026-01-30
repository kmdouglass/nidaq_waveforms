import sys
import time

import matplotlib.pyplot as plt
import nidaqmx
from nidaqmx.constants import AcquisitionType, Edge, RegenerationMode
import numpy as np


# Script configuration
MOCK_DAQ = False  # Set to True to use mock DAQ for testing without hardware
MAKE_PLOT = False  # Set to True to plot the waveform
FRAME_INTERVAL_MS = 50.0
READOUT_TIME_MS = 22.94
PARKING_FRACTION = 0.2
EXPOSURE_PP_V = 0.46
WAVEFORM_OFFSET_V = -0.075
RATE_HZ = 50000

# Hardware configuration
COUNTER = "Dev1/ctr1"
CHANNEL_CAMERA = "Dev1/ao0"
CHANNEL_GALVO = "Dev1/ao1"
SOURCE_AO = "/Dev1/Ctr1InternalOutput"
COUNTER_TRIGGER = "/Dev1/PFI3"
CAMERA_WAVEFORM_MIN_V = 0.0
CAMERA_WAVEFORM_MAX_V = 5.0
GALVO_WAVEFORM_MIN_V = -10.0
GALVO_WAVEFORM_MAX_V = 10.0


def main(
    frame_interval_ms: float = 50.0,
    readout_time_ms: float = 22.94,
    parking_fraction: float = 0.9,
    exposure_pp_V: float = 0.46,
    waveform_offset_V: float = - 0.075,
    rate_Hz: int = 50000,
    task_class=nidaqmx.Task,
):
    """Waveform generation for microscope device synchronization.

    Parameters
    ----------
    frame_interval_ms : float
        Time interval between the start of consecutive frames in milliseconds. In PVCAM rolling
        shutter mode, this is the exposure time property.
    readout_time_ms : float
        Time taken to read out a single frame in milliseconds.
    parking_fraction : float
        Fraction of the readout time during which the waveform is held at a constant "parked"
        voltage level. This should be slightly less than 1.0 to ensure the galvo has time to
        settle.
    exposure_pp_V : float
        Peak-to-peak voltage swept by the galvo during the exposure period. This determines
        the field of view and remains constant regardless of parking_fraction.
    waveform_offset_V : float
        Offset voltage of the galvo waveform in volts.
    rate_Hz : int
        Sampling rate of the waveform in Hertz.

    """
    
    #==============================================================================================
    # Validate parameters
    if frame_interval_ms <= readout_time_ms:
        raise ValueError(
            "Frame interval must be greater than readout time to ensure a non-zero exposure time."
        )
    
    if parking_fraction < 0.0 or parking_fraction > 1.0:
        raise ValueError("Parking fraction must be between 0 and 1.")
    
    #==============================================================================================
    # Calculate waveform parameters
    #   `time` refers to a duration
    waveform_period_ms = 2 * frame_interval_ms
    exposure_time_ms = frame_interval_ms - readout_time_ms
    parking_time_ms = readout_time_ms * parking_fraction
    ramp_time_ms = frame_interval_ms - parking_time_ms
    waveform_offset_ms = parking_time_ms + (readout_time_ms - parking_time_ms) / 2.0

    # Convert exposure peak-to-peak voltage to total waveform peak-to-peak voltage
    # This effectively keeps the FOV constant regardless of parking fraction.
    waveform_pp_V = exposure_pp_V * (ramp_time_ms / exposure_time_ms)

    #==============================================================================================
    # Calculate the waveform samples (must be an even number of samples)
    num_waveform_samples = int(round((waveform_period_ms / 1000.0) * rate_Hz / 2.0) * 2.0)
    num_counter_samples = num_waveform_samples // 2

    # Camera waveform
    # Pulse high for a 10% frame interval duty cycle, 5% waveform period duty cycle
    camera_waveform = np.zeros(num_waveform_samples, dtype=np.float32)
    camera_pulse_width_samples = num_waveform_samples // 20  # 5% of waveform period

    # First pulse at t=0
    camera_waveform[0:camera_pulse_width_samples] = 5.0
    camera_waveform[num_counter_samples:num_counter_samples + camera_pulse_width_samples] = 5.0

    # Galvo waveform
    parking_time_samples = int(round((parking_time_ms / 1000.0) * rate_Hz))
    ramp_time_samples = num_counter_samples - parking_time_samples
    waveform_offset_samples = int(round((waveform_offset_ms / 1000.0) * rate_Hz))

    waveform_amplitude_V = waveform_pp_V / 2.0
    waveform_high_V = waveform_offset_V + waveform_amplitude_V
    waveform_low_V = waveform_offset_V - waveform_amplitude_V
    
    if waveform_high_V > GALVO_WAVEFORM_MAX_V or waveform_low_V < GALVO_WAVEFORM_MIN_V:
        raise ValueError(
            f"Waveform voltage out of range: must be between {GALVO_WAVEFORM_MIN_V} V and {GALVO_WAVEFORM_MAX_V} V."
        )
    
    galvo_waveform = np.zeros(num_waveform_samples, dtype=np.float32)
    
    # Ramp up
    galvo_waveform[0:ramp_time_samples] = np.linspace(
        waveform_low_V,
        waveform_high_V,
        ramp_time_samples,
        endpoint=False,
    )

    # Park high
    galvo_waveform[ramp_time_samples:ramp_time_samples + parking_time_samples] = waveform_high_V

    # Ramp down
    galvo_waveform[
        num_counter_samples:num_counter_samples + ramp_time_samples
    ] = np.linspace(
        waveform_high_V,
        waveform_low_V,
        ramp_time_samples,
        endpoint=False,
    )

    # Park low
    galvo_waveform[num_counter_samples + ramp_time_samples:] = waveform_low_V

    # Apply exposure offset
    galvo_waveform = np.roll(galvo_waveform, waveform_offset_samples)

    #==============================================================================================
    # Configure and execute waveform on the DAQ device
    with task_class("CounterTask") as counter_task, task_class("AnalogOutputTask") as ao_task:
        #==========================================================================================
        # Counter config
        counter_task.co_channels.add_co_pulse_chan_freq(
            counter=COUNTER,
            freq=rate_Hz,
            duty_cycle=0.5
        )
        counter_task.timing.cfg_implicit_timing(
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=num_counter_samples
        )
        counter_task.triggers.start_trigger.cfg_dig_edge_start_trig(
            trigger_source=COUNTER_TRIGGER,
            trigger_edge=Edge.RISING
        )
        counter_task.triggers.start_trigger.retriggerable = True

        #==========================================================================================
        # Analog output config
        ao_task.ao_channels.add_ao_voltage_chan(
            CHANNEL_CAMERA,
            min_val=CAMERA_WAVEFORM_MIN_V,
            max_val=CAMERA_WAVEFORM_MAX_V
        )
        ao_task.ao_channels.add_ao_voltage_chan(
            CHANNEL_GALVO,
            min_val=GALVO_WAVEFORM_MIN_V,
            max_val=GALVO_WAVEFORM_MAX_V
        )
        ao_task.timing.cfg_samp_clk_timing(
            rate=rate_Hz,
            source=SOURCE_AO,
            sample_mode=AcquisitionType.CONTINUOUS
        )
        ao_task.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION
        ao_waveform_data = np.vstack([camera_waveform, galvo_waveform])
        ao_task.write(ao_waveform_data, auto_start=False)

        print("Press Ctrl+C to stop waveform generation...")
        try:
            # Must arm the AO task before the counter task
            ao_task.start()
            counter_task.start()

            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping waveform generation...")

    #==============================================================================================
    # Prepare results and return
    #  `start` refers to a point in time
    t = np.arange(num_waveform_samples) / rate_Hz * 1000
    exposure_start_ms = readout_time_ms

    results = {
        "time_ms": t,
        "waveform_samples_V": galvo_waveform,
        "exposure_time_ms": exposure_time_ms,
        "readout_time_ms": readout_time_ms,
        "waveform_offset_ms": waveform_offset_ms,
        "exposure_start_ms": exposure_start_ms,
        "parking_time_ms": parking_time_ms,
        "num_counter_samples": num_counter_samples,
        "rate_Hz": rate_Hz,
    }
    return results


if __name__ == "__main__":
    if MOCK_DAQ:
        from unittest.mock import MagicMock

        def make_fake_task(name=""):
            task = MagicMock()
            task.__enter__ = MagicMock(return_value=task)
            task.__exit__ = MagicMock(return_value=False)
            return task

    results = main(
        frame_interval_ms=FRAME_INTERVAL_MS,
        readout_time_ms=READOUT_TIME_MS,
        parking_fraction=PARKING_FRACTION,
        exposure_pp_V=EXPOSURE_PP_V,
        waveform_offset_V=WAVEFORM_OFFSET_V,
        rate_Hz=RATE_HZ,
        task_class=make_fake_task if MOCK_DAQ else nidaqmx.Task,
    )

    # pretty print results
    for key, value in results.items():
        print(f"{key}: {value}")

    #==============================================================================================
    # Plot the results
    if not MAKE_PLOT:
        sys.exit(0)
    
    import matplotlib
    matplotlib.use("QtAgg")

    # Double the time axis and waveform for two periods
    time_ms = np.concatenate(
        (results["time_ms"], results["time_ms"] + results["time_ms"][-1])
    )
    waveform_samples_V = np.concatenate(
        (results["waveform_samples_V"], results["waveform_samples_V"])
    )


    fig, ax = plt.subplots()
    ax.plot(time_ms, waveform_samples_V, "black")

    # Draw rectangular patches for exposure periods
    exposure_start_ms = results["exposure_start_ms"]
    exposure_end_ms = exposure_start_ms + results["exposure_time_ms"]
    for n in range(0, 2):
        ax.axvspan(
            exposure_start_ms + n * FRAME_INTERVAL_MS,
            exposure_end_ms + n * FRAME_INTERVAL_MS,
            color="blue",
            alpha=0.3,
            label="Exposure" if n == 0 else None,
        )

    # Draw rectangular patches for readout periods
    readout_start_ms = exposure_end_ms
    readout_end_ms = readout_start_ms + results["readout_time_ms"]
    for n in range(0, 2):
        ax.axvspan(
            readout_start_ms + n * FRAME_INTERVAL_MS,
            readout_end_ms + n * FRAME_INTERVAL_MS,
            color="red",
            alpha=0.3,
            label="Readout" if n == 0 else None,
        )

    # Draw vertical lines for counter pulse cycle boundaries
    counter_period_ms = results["num_counter_samples"] / results["rate_Hz"] * 1000
    num_cycles = int(np.ceil(time_ms[-1] / counter_period_ms)) + 1
    for n in range(num_cycles):
        ax.axvline(
            n * counter_period_ms,
            color="black",
            linestyle="--",
            linewidth=1,
            label="Counter Period" if n == 0 else None,
        )

    ax.set_xlim(0, 2.0 * FRAME_INTERVAL_MS + READOUT_TIME_MS)
    ax.set_xlabel("Time, ms")
    ax.set_ylabel("Voltage, V")
    ax.grid(True)
    ax.legend()

    plt.show()

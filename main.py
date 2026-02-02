import sys
import time
from typing import Any, Callable
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import nidaqmx
from nidaqmx.constants import AcquisitionType, Edge, RegenerationMode
import numpy as np


# Script configuration
MOCK_DAQ = True   # Set to True to use mock DAQ for testing without hardware
MAKE_PLOT = True  # Set to True to plot the waveform
FRAME_INTERVAL_MS = 50.0
READOUT_TIME_MS = 22.94
PARKING_FRACTION = 0.8
EXPOSURE_PP_V = 0.46
WAVEFORM_OFFSET_V = -0.075
CHANNEL_INTENSITIES = [10.0, 10.0, 10.0]
RATE_HZ = 50000

# Hardware configuration
COUNTER = "Dev1/ctr1"
CHANNEL_CAMERA = "Dev1/ao0"
CHANNEL_GALVO = "Dev1/ao1"
CHANNEL_AOTF_BLANKING = "Dev1/ao2"
CHANNEL_AOTF_CH0 = "Dev1/ao3"
CHANNEL_AOTF_CH1 = "Dev1/ao4"
CHANNEL_AOTF_CH2 = "Dev1/ao5"
CHANNEL_AOTF_CH3 = "Dev1/ao6"
SOURCE_AO = "/Dev1/Ctr1InternalOutput"
COUNTER_TRIGGER = "/Dev1/PFI3"
CAMERA_WAVEFORM_MIN_V = 0.0
CAMERA_WAVEFORM_MAX_V = 5.0
GALVO_WAVEFORM_MIN_V = -10.0
GALVO_WAVEFORM_MAX_V = 10.0
AOTF_BLANKING_V = 10.0


def main(
    channel_intensities: list[float],
    frame_interval_ms: float = 50.0,
    readout_time_ms: float = 22.94,
    parking_fraction: float = 0.8,
    exposure_pp_V: float = 0.46,
    waveform_offset_V: float = - 0.075,
    rate_Hz: int = 50000,
    task_class: Callable[[str], Any] = nidaqmx.Task,
)-> dict[str, Any]:
    """Waveform generation for microscope device synchronization.

    Parameters
    ----------
    channel_intensities : list[float]
        List of intensities (in volts) for each AOTF channel. Length must be between 1 and 4.
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

    task_class : Callable[[str], Any]
        Class or factory function to create DAQ tasks. Defaults to nidaqmx.Task.

    Returns
    -------
    dict[str, Any]
        Dictionary containing generated waveforms and related information.

    """
    
    #==============================================================================================
    # Validate parameters
    if frame_interval_ms <= readout_time_ms:
        raise ValueError(
            "Frame interval must be greater than readout time to ensure a non-zero exposure time."
        )
    
    if parking_fraction < 0.0 or parking_fraction > 1.0:
        raise ValueError("Parking fraction must be between 0 and 1.")
    
    num_channels = len(channel_intensities)
    if num_channels < 1 or num_channels > 4:
        raise ValueError("Number of channels must be between 1 and 4.")

    if any(v < 0.0 or v > 10.0 for v in channel_intensities):
        raise ValueError("Channel intensities must be between 0.0 and 10.0 V.")
    
    #==============================================================================================
    # Calculate general waveform parameters
    #   `time` refers to a duration
    waveform_interval_ms = 2 * frame_interval_ms
    exposure_time_ms = frame_interval_ms - readout_time_ms
    parking_time_ms = readout_time_ms * parking_fraction
    ramp_time_ms = frame_interval_ms - parking_time_ms
    waveform_offset_ms = parking_time_ms + (readout_time_ms - parking_time_ms) / 2.0

    # Convert exposure peak-to-peak voltage to total waveform peak-to-peak voltage
    # This effectively keeps the FOV constant regardless of parking fraction.
    waveform_pp_V = exposure_pp_V * (ramp_time_ms / exposure_time_ms)

    #==============================================================================================
    # Calculate the waveform samples (must be an even number of samples)
    num_waveform_samples = int(round((waveform_interval_ms / 1000.0) * rate_Hz / 2.0) * 2.0)
    num_counter_samples = num_waveform_samples // 2

    #----------------------------------------------------------------------------------------------
    # Camera waveform
    # Pulse high for a 10% frame interval duty cycle, 5% waveform period duty cycle
    camera_waveform = np.zeros(num_waveform_samples, dtype=np.float32)
    camera_pulse_width_samples = num_waveform_samples // 20  # 5% of waveform period

    # First pulse at t=0
    camera_waveform[0:camera_pulse_width_samples] = 5.0
    camera_waveform[num_counter_samples:num_counter_samples + camera_pulse_width_samples] = 5.0

    #----------------------------------------------------------------------------------------------
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

    #----------------------------------------------------------------------------------------------
    # AOTF waveforms
    # The AOTF waveform period is LCM(2, num_channels) frames
    if num_channels % 2 == 0:
        num_galvo_intervals = num_channels // 2
    else:
        num_galvo_intervals = num_channels
    num_aotf_frames = num_galvo_intervals * 2

    num_aotf_samples = num_galvo_intervals * num_waveform_samples
    num_readout_samples = int(round((readout_time_ms / 1000.0) * rate_Hz))

    # Tile camera and galvo waveforms to match AOTF period
    camera_waveform = np.tile(camera_waveform, num_galvo_intervals)
    galvo_waveform = np.tile(galvo_waveform, num_galvo_intervals)

    # Blanking waveform: high during every exposure
    aotf_blanking_waveform = np.zeros(num_aotf_samples, dtype=np.float32)
    for frame in range(num_aotf_frames):
        start = frame * num_counter_samples + num_readout_samples
        end = (frame + 1) * num_counter_samples
        aotf_blanking_waveform[start:end] = AOTF_BLANKING_V

    # Channel waveforms: each channel is high only during its assigned frames
    aotf_channel_waveforms = [np.zeros(num_aotf_samples, dtype=np.float32) for _ in range(4)]
    for frame in range(num_aotf_frames):
        channel = frame % num_channels
        start = frame * num_counter_samples + num_readout_samples
        end = (frame + 1) * num_counter_samples
        aotf_channel_waveforms[channel][start:end] = channel_intensities[channel]


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
        ao_task.ao_channels.add_ao_voltage_chan(CHANNEL_AOTF_BLANKING, min_val=0.0, max_val=10.0)
        ao_task.ao_channels.add_ao_voltage_chan(CHANNEL_AOTF_CH0, min_val=0.0, max_val=10.0)
        ao_task.ao_channels.add_ao_voltage_chan(CHANNEL_AOTF_CH1, min_val=0.0, max_val=10.0)
        ao_task.ao_channels.add_ao_voltage_chan(CHANNEL_AOTF_CH2, min_val=0.0, max_val=10.0)
        ao_task.ao_channels.add_ao_voltage_chan(CHANNEL_AOTF_CH3, min_val=0.0, max_val=10.0)
        ao_task.timing.cfg_samp_clk_timing(
            rate=rate_Hz,
            source=SOURCE_AO,
            sample_mode=AcquisitionType.CONTINUOUS
        )
        ao_task.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION
        ao_waveform_data = np.vstack([
            camera_waveform,
            galvo_waveform,
            aotf_blanking_waveform,
            *aotf_channel_waveforms,
        ])
        ao_task.write(ao_waveform_data, auto_start=False)

        print("Press Ctrl+C to stop waveform generation...")
        try:
            # Must arm the AO task before the counter task
            ao_task.start()
            counter_task.start()
            
            if not MOCK_DAQ:
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping waveform generation...")

    #==============================================================================================
    # Prepare results and return
    #  `start` refers to a point in time
    time_ms = np.arange(num_aotf_samples) / rate_Hz * 1000
    exposure_start_ms = readout_time_ms

    results = {
        "time_ms": time_ms,
        "galvo_waveform_V": galvo_waveform,
        "aotf_blanking_waveform_V": aotf_blanking_waveform,
        "aotf_channel_waveforms_V": aotf_channel_waveforms[:num_channels],
        "exposure_time_ms": exposure_time_ms,
        "readout_time_ms": readout_time_ms,
        "frame_interval_ms": frame_interval_ms,
        "waveform_offset_ms": waveform_offset_ms,
        "exposure_start_ms": exposure_start_ms,
        "parking_time_ms": parking_time_ms,
        "num_counter_samples": num_counter_samples,
        "num_aotf_frames": num_aotf_frames,
        "num_channels": num_channels,
        "rate_Hz": rate_Hz,
    }
    return results


if __name__ == "__main__":
    if MOCK_DAQ:
        from unittest.mock import MagicMock

        def make_fake_task(name="") -> MagicMock:
            task = MagicMock()
            task.__enter__ = MagicMock(return_value=task)
            task.__exit__ = MagicMock(return_value=False)
            return task

        task_class = make_fake_task
    else:
        task_class = nidaqmx.Task

    results = main(
        CHANNEL_INTENSITIES,
        frame_interval_ms=FRAME_INTERVAL_MS,
        readout_time_ms=READOUT_TIME_MS,
        parking_fraction=PARKING_FRACTION,
        exposure_pp_V=EXPOSURE_PP_V,
        waveform_offset_V=WAVEFORM_OFFSET_V,
        rate_Hz=RATE_HZ,
        task_class=task_class,
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

    time_ms = results["time_ms"]
    num_channels = results["num_channels"]
    num_aotf_frames = results["num_aotf_frames"]
    frame_interval_ms = results["frame_interval_ms"]
    exposure_start_ms = results["exposure_start_ms"]
    exposure_time_ms = results["exposure_time_ms"]
    readout_time_ms = results["readout_time_ms"]
    counter_period_ms = results["num_counter_samples"] / results["rate_Hz"] * 1000

    num_rows = num_channels + 2
    fig, axes = plt.subplots(num_rows, 1, sharex=True, figsize=(10, 2 * num_rows), squeeze=False)
    axes = axes.flatten()

    # Helper to draw exposure/readout periods and counter boundaries on an axis
    def draw_frame_regions(ax) -> None:
        for n in range(num_aotf_frames):
            ax.axvspan(
                exposure_start_ms + n * frame_interval_ms,
                exposure_start_ms + exposure_time_ms + n * frame_interval_ms,
                color="blue",
                alpha=0.2,
                label="Exposure" if n == 0 else None,
            )
            readout_start = exposure_start_ms + exposure_time_ms + n * frame_interval_ms
            ax.axvspan(
                readout_start,
                readout_start + readout_time_ms,
                color="red",
                alpha=0.2,
                label="Readout" if n == 0 else None,
            )
        num_cycles = int(np.ceil(time_ms[-1] / counter_period_ms)) + 1
        for n in range(num_cycles):
            ax.axvline(
                n * counter_period_ms,
                color="black",
                linestyle="--",
                linewidth=1,
                label="Frame Boundary" if n == 0 else None,
            )

    # Row 1: Galvo waveform
    ax = axes[0]
    ax.plot(time_ms, results["galvo_waveform_V"], "black")
    draw_frame_regions(ax)
    ax.set_ylabel("Galvo, V")
    ax.grid(True)
    ax.legend(loc="upper right")

    # Row 2: Blanking waveform
    ax = axes[1]
    ax.plot(time_ms, results["aotf_blanking_waveform_V"], "black")
    draw_frame_regions(ax)
    ax.set_ylabel("Blanking, V")
    ax.set_ylim(-0.5, 11)
    ax.grid(True)

    # Rows 3 to N+2: Channel waveforms
    for i, waveform in enumerate(results["aotf_channel_waveforms_V"]):
        ax = axes[2 + i]
        ax.plot(time_ms, waveform, "black")
        draw_frame_regions(ax)
        ax.set_ylabel(f"Ch{i}, V")
        ax.set_ylim(-0.5, 11)
        ax.grid(True)

    axes[-1].set_xlabel("Time, ms")
    axes[0].set_xlim(0, time_ms[-1])

    plt.tight_layout()
    plt.show()

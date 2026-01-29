import matplotlib.pyplot as plt
import nidaqmx
from nidaqmx.constants import AcquisitionType, Edge, RegenerationMode
import numpy as np


def main(
    frame_interval_ms: float = 50.0,
    readout_time_ms: float = 22.94,
    parking_fraction: float = 0.9,
    exposure_pp_V: float = 0.46,
    waveform_offset_V: float = - 0.075,
    rate_Hz: int = 50000,
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
    num_clock_samples = num_waveform_samples // 2

    parking_time_samples = int(round((parking_time_ms / 1000.0) * rate_Hz))
    ramp_time_samples = num_clock_samples - parking_time_samples
    waveform_offset_samples = int(round((waveform_offset_ms / 1000.0) * rate_Hz))

    waveform_amplitude_V = waveform_pp_V / 2.0
    waveform_high_V = waveform_offset_V + waveform_amplitude_V
    waveform_low_V = waveform_offset_V - waveform_amplitude_V
    waveform_samples = np.zeros(num_waveform_samples, dtype=np.float32)
    
    # Ramp up
    waveform_samples[0:ramp_time_samples] = np.linspace(
        waveform_low_V,
        waveform_high_V,
        ramp_time_samples,
        endpoint=False,
    )

    # Park high
    waveform_samples[ramp_time_samples:ramp_time_samples + parking_time_samples] = waveform_high_V

    # Ramp down
    waveform_samples[
        num_clock_samples:num_clock_samples + ramp_time_samples
    ] = np.linspace(
        waveform_high_V,
        waveform_low_V,
        ramp_time_samples,
        endpoint=False,
    )

    # Park low
    waveform_samples[num_clock_samples + ramp_time_samples:] = waveform_low_V

    # Apply exposure offset
    waveform_samples = np.roll(waveform_samples, waveform_offset_samples)
    
    #==============================================================================================
    # Prepare results and return
    #  `start` refers to a point in time
    t = np.arange(num_waveform_samples) / rate_Hz * 1000
    exposure_start_ms = readout_time_ms

    results = {
        "time_ms": t,
        "waveform_samples_V": waveform_samples,
        "exposure_time_ms": exposure_time_ms,
        "readout_time_ms": readout_time_ms,
        "waveform_offset_ms": waveform_offset_ms,
        "exposure_start_ms": exposure_start_ms,
        "parking_time_ms": parking_time_ms,
    }
    return results


if __name__ == "__main__":
    frame_interval_ms = 50.0
    readout_time_ms = 22.94
    parking_fraction = 0.8
    exposure_pp_V = 0.46
    waveform_offset_V = -0.075
    rate_Hz = 50000

    results = main(
        frame_interval_ms=frame_interval_ms,
        readout_time_ms=readout_time_ms,
        parking_fraction=parking_fraction,
        exposure_pp_V=exposure_pp_V,
        waveform_offset_V=waveform_offset_V,
        rate_Hz=rate_Hz,
    )

    # pretty print results
    for key, value in results.items():
        print(f"{key}: {value}")

    #==============================================================================================
    # Plot the results
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
            exposure_start_ms + n * frame_interval_ms,
            exposure_end_ms + n * frame_interval_ms,
            color="blue",
            alpha=0.3,
            label="Exposure" if n == 0 else None,
        )

    # Draw rectangular patches for readout periods
    readout_start_ms = exposure_end_ms
    readout_end_ms = readout_start_ms + results["readout_time_ms"]
    for n in range(0, 2):
        ax.axvspan(
            readout_start_ms + n * frame_interval_ms,
            readout_end_ms + n * frame_interval_ms,
            color="red",
            alpha=0.3,
            label="Readout" if n == 0 else None,
        )

    ax.set_xlim(0, 2.0 * frame_interval_ms + readout_time_ms)
    ax.set_xlabel("Time, ms")
    ax.set_ylabel("Voltage, V")
    ax.grid(True)
    ax.legend()

    plt.show()

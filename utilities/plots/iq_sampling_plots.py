import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from utilities.types.Params import FrameParams


def plot_frame(frame_params: FrameParams, zero_padded_indexes: np.ndarray = None):
    plt.figure(dpi=300)
    fig, ax = plt.subplots(figsize=(12, 6))

    # Definieren der Einheitsgrößen
    unit_width = 0.2
    unit_height = 0.5

    # Zeichnen des Synchronisationsteils
    sync_width = frame_params.num_sync_syms * unit_width
    ax.add_patch(patches.Rectangle((0, 0), sync_width, unit_height, facecolor='blue'))

    # Zeichnen des Datenteils
    data_width = frame_params.num_data_syms * unit_width
    ax.add_patch(patches.Rectangle((sync_width, 0), data_width, unit_height, facecolor='lightgray'))

    # Zeichnen der Piloten
    for i in range(frame_params.pilot_start_idx, frame_params.num_data_syms, frame_params.pilot_repetition + 1):
        pilot_x = sync_width + i * unit_width
        ax.add_patch(patches.Rectangle((pilot_x, 0), unit_width, unit_height, facecolor='red'))

    # Einfaerben der Null-Padding Symbole
    if zero_padded_indexes is not None:
        for i in zero_padded_indexes:
            zero_padding_x = sync_width + i * unit_width
            ax.add_patch(patches.Rectangle((zero_padding_x, 0), unit_width, unit_height, facecolor='black', alpha=0.3))

    # Vertikale Linien nach jedem Symbol zeichnen und Symbole annotieren
    total_symbols = frame_params.num_sync_syms + frame_params.num_data_syms
    sync_syms = frame_params.num_sync_syms

    for i in range(total_symbols + 1):
        x = i * unit_width
        ax.axvline(x=x, ymin=0.252, ymax=unit_height, color='black', linewidth=0.5, alpha=0.5)

        if i == sync_syms:
            ax.axvline(x=x, ymin=0.252, ymax=unit_height, color='black', linewidth=2, alpha=0.8)

        # Annotieren der Synchronisationssequenz
        if i < sync_syms:
            if i % 4 == 0:  # Jedes 4. Symbol in der Synchronisationssequenz
                ax.text(x + unit_width/2, 0.55, str(i), ha='center', va='top', fontsize=8)
            elif i == sync_syms - 1:
                ax.text(x + unit_width/2, 0.55, str(i), ha='center', va='top', fontsize=8)
        # Annotieren des Datenteils
        elif i >= sync_syms:
            data_sym = i - sync_syms
            if data_sym % 10 == 0:  # Jedes 10. Symbol im Datenteil
                ax.text(x + unit_width/2, -0.05, str(data_sym), ha='center', va='top', fontsize=8)

    ax.annotate(f'Synchronisation\n({sync_syms} Symbole)', (sync_width/2, -0.2), ha='center', va='top')
    ax.annotate(f"Nutzdaten\n({frame_params.num_data_syms} Symbole; Pattern: Ab Index {frame_params.pilot_start_idx} ein Pilot gefolgt von {frame_params.pilot_repetition} Datensymbolen)", (sync_width + data_width / 2, -0.2), ha='center', va='top')

    legend_elements = [
        patches.Patch(facecolor='blue', edgecolor='black', label='Synchronisation'),
        patches.Patch(facecolor='lightgray', edgecolor='black', label='Daten'),
        patches.Patch(facecolor='red', edgecolor='black', label='Piloten')
    ]

    if zero_padded_indexes is not None:
        legend_elements.append(patches.Patch(facecolor='black', edgecolor='black', alpha=0.3, label='Zero-Padding'))

    ax.legend(handles=legend_elements, loc='upper right')

    ax.set_xlim(0, sync_width + data_width)
    ax.set_ylim(-0.5, 1.5)
    ax.axis('off')

    plt.title(f'Darstellung eines Frames ({total_symbols} Symbole) mit Synchronisationssequenz,\nPiloten- und Daten-Symbolen', pad=20)
    plt.tight_layout()

    plt.show()

def plot_sampling_parameter():
    T_s = 1  # Symboldauer (von -0.5 bis 0.5)
    num_samples = 3  # Anzahl der Samples innerhalb des Pulses
    T_sample = T_s / (num_samples - 1)  # Abstand zwischen den Samples

    T = np.linspace(-2, 2, 1000)
    pulse = np.where((T >= -T_s / 2) & (T <= T_s / 2), 1, 0)

    samples_x = np.linspace(-T_s / 2, T_s / 2, num_samples)
    samples_y = [1] * num_samples

    offset_samples_x = np.array([-2, -1.5, -1, 1, 1.5, 2])
    offset_samples_y = [0] * len(offset_samples_x)

    plt.figure(figsize=(10, 5))

    plt.plot(T, pulse, label=r'$\rm{rect}(\frac{t}{T_s})$', color='blue', linewidth=2)
    plt.fill_between(T, pulse, alpha=0.3)

    plt.scatter(samples_x, samples_y, color='black', label='Samples (innerhalb)', zorder=5)

    plt.scatter(offset_samples_x, offset_samples_y, color='white', edgecolor='black', label='Samples (außerhalb)', zorder=5)

    plt.annotate("", xy=(-T_s / 2, 0.85), xytext=(T_s / 2, 0.85), arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    plt.text(0, 0.9, r'$T_s$', color='red', fontsize=12, ha='center')

    for i in range(num_samples - 1):
        plt.annotate("", xy=(samples_x[i], 0.7), xytext=(samples_x[i + 1], 0.7),
                     arrowprops=dict(arrowstyle='<->', color='green', lw=2))
        plt.text((samples_x[i] + samples_x[i + 1]) / 2, 0.75, r'$T_{\rm{Sample}}$', color='green', fontsize=10, ha='center')

    plt.title(r'Parameter am Beispiel eines $\rm{rect}(\frac{t}{T_s})$: num_tabs$=$'+f'{int(samples_x.size + offset_samples_x.size)}' + r", sps$=$" + f"{samples_x.size}" + r", $T_s = $"+ f"{T_s} s," + r" $T_{\rm{Sample}} = $ " + f"{T_sample} s")
    plt.xlabel('t [s]')
    plt.ylabel('Amplitude')
    plt.ylim(-0.2, 1.2)
    plt.xlim(-2.1, 2.1)
    plt.grid()
    plt.legend()

    plt.show()

def plot_iq_samples(x=None, x_i=None, x_q=None, sps=None, marked_symbol=5):
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    if sps is None:
        index = marked_symbol - 1
    else:
        index = (marked_symbol - 1) * sps

    if (x_i is None) and (x_q is None) and (x is not None):
        x_i = np.real(x)
        x_q = np.imag(x)

    data = [x_i, x_q]
    titles = [r"Realteil: $ x_{\rm{I}}[n] $", r"Imaginärteil: $ x_{\rm{Q}}[n]$"]
    ylabels = [r'$x_{\rm{I}}[n]$', r'$x_{\rm{Q}}[n]$']

    for i, ax in enumerate(axes[:2]):
        ax.stem(data[i])
        ax.grid(True)
        ax.set_xlabel(r'$n$')
        ax.set_ylabel(ylabels[i])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_title(titles[i])
        ax.plot(index, data[i][index], 'o', color='orange')

    # Scatter plot
    if sps is None:
        axes[2].scatter(x_i[::sps], x_q[::sps])
    else:
        axes[2].scatter(x_i, x_q)
    axes[2].plot(x_i[index], x_q[index], 'o', color='orange')
    axes[2].set_aspect('equal', adjustable='box')
    axes[2].grid(True)

    # Draw unit circle
    unit_circle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='--')
    axes[2].add_artist(unit_circle)

    # Set titles and labels
    axes[2].set_title('IQ-Plot')
    axes[2].set_xlabel(r'$\Im$')
    axes[2].set_ylabel(r'$\Re$')

    plt.tight_layout()
    return fig, axes


def plot_shaped_signals(
    h_pulse_form, x_i_no_shape, x_i_shaped, x_i_shaped_spectrum,
    x_q_no_shape, x_q_shaped, x_q_shaped_spectrum, t, f):
    """
    Plotte die geformten Basisbandsignale und ihre Spektren.

    :param h_pulse_form: Instanz der Pulse-Form
    :param x_i_no_shape: Ungeformtes Signal (I-Komponente)
    :param x_i_shaped: Geformtes Signal (I-Komponente)
    :param x_i_shaped_spectrum: Spektrum des geformten Signals (I-Komponente)
    :param x_q_no_shape: Ungeformtes Signal (Q-Komponente)
    :param x_q_shaped: Geformtes Signal (Q-Komponente)
    :param x_q_shaped_spectrum: Spektrum des geformten Signals (Q-Komponente)
    :param t: Zeitachse
    :param f: Frequenzachse
    """
    # Generiere das geformte Pulssignal
    h_pulse = h_pulse_form.generate_pulse()

    # Erstelle die Plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Plot I-Komponente
    dirac_sum_zero_padded_i = np.zeros(len(x_i_shaped))
    dirac_sum_zero_padded_i[h_pulse.size//2-1:h_pulse.size//2+len(x_i_no_shape)-1] = x_i_no_shape
    ax1.stem(t, dirac_sum_zero_padded_i, 'r')
    ax1.plot(t, x_i_shaped, 'b', linewidth=4, alpha=0.9)
    ax1.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax1.grid(True)
    ax1.set_xlabel(r'$t$ [s]')
    ax1.set_ylabel(r'$x_{\rm{I, shaped}}[n]$')
    ax1.set_title(r"Geformtes Basisbandsignal: $x_{\rm{I, shaped}}[n]$")

    # Plot Spektrum I-Komponente
    ax2.plot(f, 10*np.log10(np.abs(x_i_shaped_spectrum)**2))
    ax2.set_xlabel(r"$f$ [Hz]")
    ax2.set_ylabel(r"$|X_{\rm{I, shaped}}(f)|$ [dB]")
    ax2.set_title(r"Spektrum des geformten Basisbandsignals: $X_{\rm{I, shaped}}(f)$")
    ax2.grid(True)
    max_value_i = np.max(10*np.log10(np.abs(x_i_shaped_spectrum)**2))
    ax2.set_ylim([max_value_i-100, max_value_i*1.25])

    # Plot Q-Komponente
    dirac_sum_zero_padded_q = np.zeros(len(x_q_shaped))
    dirac_sum_zero_padded_q[h_pulse.size//2-1:h_pulse.size//2+len(x_q_no_shape)-1] = x_q_no_shape
    ax3.stem(t, dirac_sum_zero_padded_q, 'r')
    ax3.plot(t, x_q_shaped, 'b', linewidth=4, alpha=0.9)
    ax3.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax3.grid(True)
    ax3.set_xlabel(r'$t$ [s]')
    ax3.set_ylabel(r'$x_{\rm{Q, shaped}}[n]$')
    ax3.set_title(r"Geformtes Basisbandsignal: $x_{\rm{Q, shaped}}[n]$")

    # Plot Spektrum Q-Komponente
    tmp = x_q_shaped_spectrum + 1e-100 if np.all(x_q_shaped_spectrum == 0) else x_q_shaped_spectrum
    ax4.plot(f, 10*np.log10(np.abs(tmp)**2))
    ax4.set_xlabel(r"$f$ [Hz]")
    ax4.set_ylabel(r"$|X_{\rm{Q, shaped}}(f)|$ [dB]")
    ax4.set_title(r"Spektrum des geformten Basisbandsignals: $X_{\rm{Q, shaped}}(f)$")
    ax4.grid(True)
    max_value_q = np.max(10*np.log10(np.abs(tmp)**2))
    if max_value_q > 0:
        ax4.set_ylim([-max_value_q, max_value_q*1.25])
    else:
        ax4.set_ylim([max_value_q-100, 5])

    plt.tight_layout()
    plt.show()

def plot_carrier_signals(
        t: np.ndarray, s_c_I: np.ndarray, s_c_Q: np.ndarray, f_c: float):
    """
    Plotte die Trägersignale und einen Zoom auf zwei Perioden.

    :param t: Zeitachse
    :param s_c_I: In-Phase-Komponente des Trägersignals
    :param s_c_Q: Quadratur-Komponente des Trägersignals
    :param f_c: Trägerfrequenz
    """
    # Erstelle die Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot der vollständigen Trägersignale
    ax1.plot(t, s_c_I)
    ax1.plot(t, s_c_Q)
    ax1.set_title(r"Träger $s_c(t)$")
    ax1.set_xlabel(r"$t$ [s]")
    ax1.set_ylabel(r"$s_c(t)$")
    ax1.legend([r"$s_{\rm{c,I}}(t) $", r"$s_{\rm{c,Q}}(t) $"])
    ax1.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax1.grid(True)

    # Plot eines Zooms auf zwei Perioden
    period = 1 / f_c
    zoom_start = 0
    zoom_end = 2 * period
    ax2.plot(t, s_c_I)
    ax2.plot(t, s_c_Q)
    ax2.set_xlim([zoom_start, zoom_end])
    ax2.set_title(r"Träger Zoom: $s_c(t)$")
    ax2.set_xlabel(r"$t$ [s]")
    ax2.set_ylabel(r"$s_c(t)$")
    ax2.legend([r"$s_{\rm{c,I}}(t) $", r"$s_{\rm{c,Q}}(t) $"])
    ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_passband_signals(
    t: np.ndarray, x_i_shaped: np.ndarray, s_c_I: np.ndarray,
    x_q_shaped: np.ndarray, s_c_Q: np.ndarray, s_hf: np.ndarray
):
    """
    Plotte die Passbandsignale.

    :param t: Zeitachse
    :param x_i_shaped: Geformtes Signal (I-Komponente)
    :param s_c_I: In-Phase-Komponente des Trägersignals
    :param x_q_shaped: Geformtes Signal (Q-Komponente)
    :param s_c_Q: Quadratur-Komponente des Trägersignals
    :param s_hf: Hochfrequenzsignal
    """
    # Erstelle die Plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))

    # Plot I(t) * s_c,I(t)
    ax1.plot(t, x_i_shaped * s_c_I, label=r"$x_{\rm{I, shaped}}(t) $ $ s_{\rm{c,I}}(t)$")
    ax1.plot(t, x_i_shaped, color='red', label=r"$x_{\rm{I, shaped}}(t) $")
    ax1.set_title(r"Passbandsignal: $x_{\rm{I, shaped}}(t) $ $ s_{\rm{c,I}}(t)$")
    ax1.set_ylabel(r"$x_{\rm{I, shaped}}(t) s_{\rm{c,I}}(t)$")
    ax1.set_xlabel(r"$t$ [s]")
    ax1.legend()
    ax1.grid(True)

    # Plot Q(t) * s_c,Q(t)
    ax2.plot(t, x_q_shaped * s_c_Q, label=r"$x_{\rm{Q, shaped}}(t) $ $ s_{\rm{c,Q}}(t)$")
    ax2.plot(t, x_q_shaped, color='red', label=r"$x_{\rm{Q, shaped}}(t) $")
    ax2.set_title(r"Passbandsignal: $x_{\rm{Q, shaped}}(t) $ $ s_{\rm{c,Q}}(t)$")
    ax2.set_ylabel(r"$x_{\rm{Q, shaped}}(t) s_{\rm{c,Q}}(t)$")
    ax2.set_xlabel(r"$t$ [s]")
    ax2.legend()
    ax2.grid(True)

    # Plot s_HF(t)
    ax3.plot(t, s_hf)
    ax3.set_title(r"Passbandsignal: $s_{\rm{HF}}(t)$")
    ax3.set_ylabel(r"$s_{\rm{HF}}(t)$")
    ax3.set_xlabel(r"$t$ [s]")
    ax3.grid(True)

    plt.tight_layout()
    plt.show()


def plot_passband_spectrum(
        f: np.ndarray, s_hf_spectrum: np.ndarray, fc: float
):
    """
    Plotte das Spektrum des Passbandsignals.

    :param f: Frequenzachse
    :param s_hf_spectrum: Spektrum des Passbandsignals
    """
    # Plot des Spektrums
    plt.suptitle(r"Spektrum des Passbandsignals: $S_{\rm{HF}}(f)$ mit $f_c = $ " + f"{fc} Hz")
    plt.plot(f, 10 * np.log10(np.abs(s_hf_spectrum) ** 2))
    plt.xlabel(r"$f$ [Hz]")
    plt.ylabel(r"$| S_{\rm{HF}}(f) |$ [dB]")
    plt.grid(True)

    # Setze die y-Achsenbegrenzung
    max_val = np.max(10 * np.log10(np.abs(s_hf_spectrum) ** 2))
    plt.ylim(max_val - 100, 1.25 * max_val)
    plt.xlim(-2 * fc, 2 * fc)
    plt.show()

def plot_downmixed_signals(
    t: np.ndarray, hat_i_no_filter: np.ndarray, hat_i_no_filter_spectrum: np.ndarray,
    hat_q_no_filter: np.ndarray, hat_q_no_filter_spectrum: np.ndarray, f: np.ndarray, fc: float
):
    """
    Plotte die heruntergemischten Basisbandsignale und ihre Spektren.

    :param t: Zeitachse
    :param hat_i_no_filter: Heruntergemischtes Signal (I-Komponente, ohne Filter)
    :param hat_i_no_filter_spectrum: Spektrum des heruntergemischten Signals (I-Komponente, ohne Filter)
    :param hat_q_no_filter: Heruntergemischtes Signal (Q-Komponente, ohne Filter)
    :param hat_q_no_filter_spectrum: Spektrum des heruntergemischten Signals (Q-Komponente, ohne Filter)
    :param f: Frequenzachse
    """
    # Erstelle die Plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot I-Komponente im Zeitbereich
    ax1.plot(t, hat_i_no_filter)
    ax1.set_title(r"Heruntergemischtes Basisbandsignal: $\hat{I}_{\rm{No Filter}}(t)$")
    ax1.set_xlabel(r"$t$ [s]")
    ax1.set_ylabel(r"$\hat{I}_{\rm{No Filter}}(t)$")
    ax1.grid(True)
    #ax1.set_yticks([-1, -0.5, 0, 0.5, 1])
    y_max = np.max(np.abs(hat_i_no_filter))
    ax1.set_ylim(-1.25*y_max, 1.25*y_max)

    # Plot I-Komponente im Frequenzbereich
    ax2.plot(f, 10 * np.log10(np.abs(hat_i_no_filter_spectrum) ** 2))
    ax2.set_title(r"Spektrum des heruntergemischten Signals: $\hat{I}_{\rm{No Filter}}(f)$")
    ax2.grid(True)
    ax2.set_xlabel(r"$f$ [Hz]")
    ax2.set_ylabel(r"|$\hat{I}_{\rm{No Filter}}(f)$| [dB]")
    max_val_i = np.max(10 * np.log10(np.abs(hat_i_no_filter_spectrum) ** 2))
    ax2.set_ylim(max_val_i-100, 1.25*max_val_i)
    ax2.set_xlim(-2 * fc * 1.5, 2 * fc *1.5)

    # Plot Q-Komponente im Zeitbereich
    ax3.plot(t, hat_q_no_filter)
    ax3.set_title(r"Heruntergemischtes Basisbandsignal: $\hat{Q}_{\rm{No Filter}}(t)$")
    ax3.set_xlabel(r"$t$ [s]")
    ax3.set_ylabel(r"$\hat{Q}_{\rm{No Filter}}(t)$")
    ax3.grid(True)
    #ax3.set_yticks([-1, -0.5, 0, 0.5, 1])
    y_max = np.max(np.abs(hat_q_no_filter))
    ax3.set_ylim(-1.25*y_max, 1.25*y_max)

    # Plot Q-Komponente im Frequenzbereich
    ax4.plot(f, 10 * np.log10(np.abs(hat_q_no_filter_spectrum) ** 2))
    ax4.set_title(r"Spektrum des heruntergemischten Signals: $\hat{Q}_{\rm{No Filter}}(f)$")
    ax4.grid(True)
    ax4.set_xlabel(r"$f$ [Hz]")
    ax4.set_ylabel(r"|$\hat{Q}_{\rm{No Filter}}(f)$| [dB]")
    max_val_q = np.max(10 * np.log10(np.abs(hat_q_no_filter_spectrum) ** 2))
    ax4.set_ylim(max_val_q-100, 1.25*max_val_q)
    ax4.set_xlim(-2 * fc * 1.5, 2 * fc * 1.5)

    plt.tight_layout()
    plt.show()

def plot_lowpass_filter(
    f: np.ndarray, H_tp: np.ndarray, border_freq: float
):
    """
    Plotte die Übertragungsfunktion des Tiefpassfilters.

    :param f: Frequenzachse
    :param H_tp: Übertragungsfunktion des Tiefpassfilters
    """
    # Plot der Übertragungsfunktion
    plt.suptitle(r"Tiefpassfilter: $H_{\rm{TP}}(f)$")
    plt.plot(f, H_tp)
    plt.xlabel(r"$f$ [Hz]")
    plt.ylabel(r"$|H_{\rm{TP}}(f)|$")
    plt.xlim(-10 * border_freq, 10 * border_freq)
    plt.grid(True)

def plot_filtered_spectra(
    f: np.ndarray, hat_i_filtered_spectrum: np.ndarray, hat_q_filtered_spectrum: np.ndarray
):
    """
    Plotte die Spektren der heruntergemischten und gefilterten Signale.

    :param f: Frequenzachse
    :param hat_i_filtered_spectrum: Spektrum des gefilterten Signals (I-Komponente)
    :param hat_q_filtered_spectrum: Spektrum des gefilterten Signals (Q-Komponente)
    """
    # Erstelle die Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot I-Komponente
    ax1.plot(f, 10 * np.log10(np.abs(hat_i_filtered_spectrum) ** 2))
    ax1.set_title(r"Heruntergemischtes TP gefiltertes Signal: $\hat{I}_{\rm{Filtered}}(f)$")
    ax1.set_xlabel(r"$f$ [Hz]")
    ax1.set_ylabel(r"| $\hat{I}_{\rm{Filtered}}(f) | $ [dB]")
    ax1.grid(True)
    max_val_i = np.max(10 * np.log10(np.abs(hat_i_filtered_spectrum) ** 2))
    ax1.set_ylim(-max_val_i, 1.25*max_val_i)

    # Plot Q-Komponente
    ax2.plot(f, 10 * np.log10(np.abs(hat_q_filtered_spectrum) ** 2))
    ax2.set_title(r"Heruntergemischtes TP gefiltertes Signal: $\hat{Q}_{\rm{Filtered}}(f)$")
    ax2.set_xlabel(r"$f$ [Hz]")
    ax2.set_ylabel(r"| $\hat{Q}_{\rm{Filtered}}(f) | $ [dB]")
    ax2.grid(True)
    max_val_q = np.max(10 * np.log10(np.abs(hat_q_filtered_spectrum) ** 2))
    ax2.set_ylim(-max_val_q, 1.25*max_val_q)

    plt.tight_layout()
    plt.show()



def plot_filtered_signals(
    t: np.ndarray, hat_i_filtered: np.ndarray, x_i_shaped: np.ndarray,
    hat_q_filtered: np.ndarray, x_q_shaped: np.ndarray, params, symbols: np.ndarray
):
    """
    Plotte die gefilterten und heruntergemischten Empfangssignale.

    :param t: Zeitachse
    :param hat_i_filtered: Gefiltertes Signal (I-Komponente)
    :param x_i_shaped: Geformtes Signal (I-Komponente)
    :param hat_q_filtered: Gefiltertes Signal (Q-Komponente)
    :param x_q_shaped: Geformtes Signal (Q-Komponente)
    :param params: Parameterobjekt mit Basisbandinformationen
    :param symbols: Array der Symbole (nicht direkt verwendet, aber in der ursprünglichen Berechnung vorhanden)
    """
    # Erstelle die Plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))

    # Plot I-Komponente
    ax1.plot(t, hat_i_filtered.real)
    ax1.plot(t, x_i_shaped)
    ax1.grid(True)
    ax1.legend([r'$\hat{I}_{\rm{Filtered}}(t)$', r'$x_{\rm{I,shaped}}(t)$'])

    # Markiere Abtastzeitpunkte
    start = params.baseband.num_samps // 2 + 1
    end = hat_i_filtered.size - params.baseband.num_samps // 2
    n = start
    while n <= end:
        ax1.plot(t[n], hat_i_filtered[n].real, 'o')
        n += params.baseband.sps

    ax1.set_xlabel(r't [s]')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(r'Gefiltertes und heruntergemischtes Empfangssignal: $\hat{I}_{\rm{Filtered}}(t)$')

    # Plot Q-Komponente
    ax2.plot(t, hat_q_filtered)
    ax2.plot(t, x_q_shaped)
    ax2.grid(True)
    ax2.legend([r'$\hat{Q}_{\rm{Filtered}}(t)$', r'$x_{\rm{Q,shaped}}(t)$'])

    # Markiere Abtastzeitpunkte
    start = params.baseband.num_samps // 2 + 1
    end = hat_q_filtered.size - params.baseband.num_samps // 2
    n = start
    while n <= end:
        ax2.plot(t[n], hat_q_filtered[n], 'o')
        n += params.baseband.sps

    ax2.set_xlabel(r't [s]')
    ax2.set_ylabel('Amplitude')
    ax2.set_title(r'Gefiltertes und heruntergemischtes Empfangssignal: $\hat{Q}_{\rm{Filtered}}(t)$')

    plt.tight_layout()
    plt.show()


def plot_signals(
    t: np.ndarray, x_i_shaped: np.ndarray, x_q_shaped: np.ndarray,
    s_hf: np.ndarray, hat_i_filtered: np.ndarray, hat_q_filtered: np.ndarray
):
    """
    Plotte die geformten Basisbandsignale, das Passbandsignal und die gefilterten Basisbandsignale.

    :param t: Zeitachse
    :param x_i_shaped: Geformtes Signal (I-Komponente)
    :param x_q_shaped: Geformtes Signal (Q-Komponente)
    :param s_hf: Passbandsignal
    :param hat_i_filtered: Gefiltertes Signal (I-Komponente)
    :param hat_q_filtered: Gefiltertes Signal (Q-Komponente)
    """
    # Erstelle die Plots
    plt.figure(figsize=(12, 8))

    # Plot geformte Basisbandsignale
    plt.subplot(3, 1, 1)
    plt.plot(t, x_i_shaped)
    plt.plot(t, x_q_shaped)
    plt.title(r"Geformtes Basisbandsignal")
    plt.xlabel(r'$t$ [s]')
    plt.ylabel(r"$x_{\rm{I,Q, shaped}}(t)$")
    plt.legend([r"$x_{\rm{I, shaped}}(t)$", r"$x_{\rm{Q, shaped}}(t)$"])
    plt.grid(True)

    # Plot Passbandsignal
    plt.subplot(3, 1, 2)
    plt.plot(t, s_hf)
    plt.title(r"Passbandsignal: $s_{\rm{HF}}(t)$")
    plt.xlabel(r'$t$ [s]')
    plt.ylabel(r"$s_{\rm{HF}}(t)$")
    plt.grid(True)

    # Plot gefilterte Basisbandsignale
    plt.subplot(3, 1, 3)
    plt.plot(t, hat_i_filtered)
    plt.plot(t, hat_q_filtered)
    plt.title(r"Gefiltertes Signal (Basisband)")
    plt.xlabel(r'$t$ [s]')
    plt.ylabel(r"Amplitude")
    plt.legend([r"$\hat{I}_{\rm{Filtered}}(t)$", r"$\hat{Q}_{\rm{Filtered}}(t)$"])
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_spectra(
        f: np.ndarray, baseband_signal_normalized: np.ndarray,
        x_i_shaped_normalized: np.ndarray, hat_i_filtered_normalized: np.ndarray
):
    """
    Plotte die Spektren der normalisierten Signale.

    :param f: Frequenzachse
    :param baseband_signal_normalized: Normalisiertes Basisbandsignal
    :param x_i_shaped_normalized: Normalisiertes geformtes Signal (I-Komponente)
    :param hat_i_filtered_normalized: Normalisiertes gefiltertes Signal (I-Komponente)
    """
    # Plot der Spektren
    plt.plot(f, np.fft.fftshift(10 * np.log10(np.abs(np.fft.fft(baseband_signal_normalized)) ** 2)), alpha=0.5,
             linestyle='--')
    plt.plot(f, np.fft.fftshift(10 * np.log10(np.abs(np.fft.fft(x_i_shaped_normalized)) ** 2)), alpha=0.5,
             linestyle='-')
    plt.plot(f, np.fft.fftshift(10 * np.log10(np.abs(np.fft.fft(hat_i_filtered_normalized)) ** 2)), color='black',
             linewidth=3, alpha=0.75)

    plt.legend([r"$H_{\rm{TP}}(f)$", r"$X_{\rm{I,shaped}}(f)$", r"$\hat{I}_{\rm{shaped}}(f)$"])
    plt.xlabel(r"$f$ [Hz]")
    plt.ylabel("[dB]")

    # Setze die y-Achsenbegrenzung
    max_value = np.max(10 * np.log10(np.abs(np.fft.fft(hat_i_filtered_normalized)) ** 2))
    plt.ylim(-4*max_value, 2 * max_value)

    plt.grid(True)
    plt.show()


def plot_iq_symbols(
    data_symbols_received: np.ndarray, data_symbols_received_eq: np.ndarray,
    params, pilot_start_idx: int, pilot_symbol_repetition: int
):
    """
    Plotte die IQ-Diagramme der empfangenen Symbole.

    :param data_symbols_received: Empfangene Symbole (unequalisiert)
    :param data_symbols_received_eq: Empfangene Symbole (equalisiert)
    :param params: Parameterobjekt mit Frame-Informationen
    :param pilot_start_idx: Startindex der Pilotsymbole
    :param pilot_symbol_repetition: Wiederholungsrate der Pilotsymbole
    """
    # Erstelle die Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('IQ-Plots of Received Symbols', fontsize=16)

    def create_iq_plot(ax, data, title, pilot_start_idx, pilot_symbol_repetition):
        max_val = np.max(np.abs(data))
        scale_factor = 1.25

        # Plot all symbols in red
        ax.scatter(np.real(data), np.imag(data), alpha=0.5, color='red', label='Data Symbols')

        # Calculate pilot indices and plot them in blue
        pilot_indices = np.arange(pilot_start_idx, len(data), pilot_symbol_repetition+1)
        ax.scatter(np.real(data[pilot_indices]), np.imag(data[pilot_indices]), alpha=0.5, color='blue', label='Pilot Symbols')

        # Set equal aspect ratio to make the plot square
        ax.set_aspect('equal', 'box')

        # Set limits to be square and symmetric
        limit = scale_factor * max_val
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)

        ax.grid(True)
        ax.set_xlabel(r'$\Re$', fontsize=14)
        ax.set_ylabel(r'$\Im$', fontsize=14)
        ax.set_title(title, fontsize=14)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)

        # Add legend
        ax.legend()

    # Erstelle die IQ-Plots
    create_iq_plot(ax1, data_symbols_received, 'Not Equalized Symbols', pilot_start_idx, pilot_symbol_repetition)
    create_iq_plot(ax2, data_symbols_received_eq, 'Equalized Symbols', pilot_start_idx, pilot_symbol_repetition)

    plt.tight_layout()
    plt.show()

# Beispielaufruf
#


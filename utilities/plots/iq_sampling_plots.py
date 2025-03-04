import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_frame(params):
    plt.figure(dpi=300)
    fig, ax = plt.subplots(figsize=(12, 6))

    # Definieren der Einheitsgrößen
    unit_width = 0.2
    unit_height = 0.5

    # Zeichnen des Synchronisationsteils
    sync_width = params['frame']['sync_syms'] * unit_width
    ax.add_patch(patches.Rectangle((0, 0), sync_width, unit_height, facecolor='blue'))

    # Zeichnen des Datenteils
    data_width = params['frame']['data_syms'] * unit_width
    ax.add_patch(patches.Rectangle((sync_width, 0), data_width, unit_height, facecolor='lightgray'))

    # Zeichnen der Piloten
    for i in range(params['frame']['pilot_start_idx'], params['frame']['data_syms'], params['frame']['pilot_repetition']+1):
        pilot_x = sync_width + i * unit_width
        ax.add_patch(patches.Rectangle((pilot_x, 0), unit_width, unit_height, facecolor='red'))

    # Vertikale Linien nach jedem Symbol zeichnen und Symbole annotieren
    total_symbols = params['frame']['sync_syms'] + params['frame']['data_syms']
    sync_syms = params['frame']['sync_syms']

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
    ax.annotate(f"Nutzdaten\n({params['frame']['data_syms']} Symbole; Pattern: Ab Index {params['frame']['pilot_start_idx']} ein Pilot gefolgt von {params['frame']['pilot_repetition']} Datensymbolen)", (sync_width + data_width/2, -0.2), ha='center', va='top')

    legend_elements = [
        patches.Patch(facecolor='blue', edgecolor='black', label='Synchronisation'),
        patches.Patch(facecolor='lightgray', edgecolor='black', label='Daten'),
        patches.Patch(facecolor='red', edgecolor='black', label='Piloten')
    ]
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
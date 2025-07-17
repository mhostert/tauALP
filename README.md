# $\tau$ALP

This repo contains the work for the following publication:

## Citation

@article{


}

## Description

* `0_generate_tau_events.ipynb`: generates dataframes of tau events from existing Pythia8 simulation files. Run the first few cells to generate `.parquet` files of tau events for faster event rate evaluation. To include less Pythia events, simply replace:
    ```
    NUMI_files = ['pythia8_events/soft_120_GeV',
                'pythia8_events/soft_120_GeV_3e3',
                'pythia8_events/soft_120_GeV_2e4']
    ```
    by, for example,

    ```
    NUMI_files = ['pythia8_events/soft_120_GeV']
    ```
    or
    ```
    NUMI_files = 'pythia8_events/soft_120_GeV_0'
    ```

* `generate_rate_tables.py`: this is where the event rate sensitivities are calculated with a parameter scan. Note that this is quite a slow and memory-intensive task. You can reduce the number of events used for lighter and faster evaluation (see above how to regenerate new `.parquet` files or simply pass the pythia files to     `Experiment` class).


* `1_plot_kinematics.ipynb`: self-explanatory.

* `2_plot_alp_properties.ipynb`: self-explanatory.

* `3_lfv_main_plots.ipynb`: LFV event-rate sensitivities using the results of the parameter scan.

* `4_lfc_main_plots.ipynb`: LFC event-rate sensitivities using the results of the parameter scan.

* `5_lfv_uncertainty_study.ipynb`: some tests and studies with the simplified approach to generate taus.

The Pythia event generation is performed in `.cpp` and `generate_taus.py` files.
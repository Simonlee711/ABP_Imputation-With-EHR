# Plot generation 

To generate plots, 3 steps need to be completed: 

1. Generate predictions using [predict.py](models/predict.py)
2. Create data for plots using [get_plot_data.py](visualization/get_plot_data.py)
3. Run plot generation script [make_plots.py](visualization/make_plots.py)

The first step uses a trained model to generate imputed waveforms using the observed signals. 
The predicted waveforms, along with the true waveforms, the input signals used as input to
the model, and metadata about the windows are saved to .csv files. 
These .csv files are then used as input to [get_plot_data.py](visualization/get_plot_data.py) 
which calculates various metrics. These metrics are then written to files, which are then read by 
[make_plots.py](visualization/make_plots.py) to create the plots and tables. 


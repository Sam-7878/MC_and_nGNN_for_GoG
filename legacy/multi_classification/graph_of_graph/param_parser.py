import argparse

def parameter_parser():
    """
    Parses command line parameters with options tailored for different graph models.
    """
    parser = argparse.ArgumentParser(description="Run Graph Models")
    
    # Common arguments for all models
    parser.add_argument("--model", type=str, default = "SEAL", choices=["DVGGA", "GOGNN", "SEAL"], 
                        help="Type of the graph model to run.")
    parser.add_argument("--graphs",
                        nargs="?",
                        default="../../../_data/GoG/polygon/",
	                help="Folder with training graph jsons.")

    parser.add_argument("--hierarchical-graph",
                        nargs="?",
                        default="../../../_data/GoG/polygon/edges/global_edges.csv",
	                help="Hierarchical edge list.")
                    
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate.")
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=5e-5, help="Weight decay for Adam optimizer.")
    parser.add_argument("--gamma", type=float, default=1e-5, help="Regularization coefficient for model.")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout ratio.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of data to use for training.")
    parser.add_argument("--device", default="cuda:0", help="Device to run the model on (e.g., cpu, cuda:0).")
    parser.add_argument("--chain", default="polygon", help="Blockchain")
    parser.add_argument("--split_type", type=str, default="temporal", choices=["random", "temporal"])
    
    # Model-specific arguments
    parser.add_argument("--first-gcn-dimensions", type=int, default=16,
                        help="Number of filters in the first graph convolution layer.")
    parser.add_argument("--second-gcn-dimensions", type=int, default=16,
                        help="Number of filters in the second graph convolution layer.")
    parser.add_argument("--first-dense-neurons", type=int, default=16,
                        help="Number of neurons in the first dense layer.")
    parser.add_argument("--second-dense-neurons", type=int, default=8,
                        help="Number of neurons in the second dense layer.")
    parser.add_argument("--macro-gcn-dimensions", type=int, default=16,
                        help="Number of filters in the macro graph convolution layer.")

    parser.add_argument("--vgae-hidden-dimensions", type=int, default=8,
                        help="Hidden dimensions for variational graph autoencoder models, applicable if model_type is DVGGA.")

    parser.add_argument("--nhid", type=int, default=32,
                        help="Number of hidden units in specific layers, applicable if model_type is GOGNN.")

    parser.add_argument("--ddi-nhid", type=int, default=32,
                        help="Specific hidden dimensions for drug-drug interaction layers, applicable if model_type is GOGNN.")

    parser.add_argument("--pooling-ratio", type=float, default=0.6,
                        help="Graph pooling ratio, applicable if model_type is GOGNN.")

    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training, applicable if model_type is GOGNN.")

    return parser.parse_args()


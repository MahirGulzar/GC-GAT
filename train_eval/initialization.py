# Import datasets
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from datasets.interface import TrajectoryDataset
from datasets.nuScenes.nuScenes_raster import NuScenesRaster
from datasets.nuScenes.nuScenes_vector import NuScenesVector
from datasets.nuScenes.nuScenes_graphs import NuScenesGraphs

# Import models
from models.model import PredictionModel
from models.encoders.raster_encoder import RasterEncoder
from models.encoders.polyline_subgraph import PolylineSubgraphs
from models.encoders.pgp_encoder import PGPEncoder
from models.encoders.st_encoder import STEncoder
from models.encoders.pgp_mod_encoder import PGPModEncoder
from models.aggregators.concat import Concat
from models.aggregators.global_attention import GlobalAttention
from models.aggregators.goal_conditioned import GoalConditioned
from models.aggregators.pgp import PGP
from models.aggregators.pass_through import PassThrough
from models.aggregators.ac_aggregator import ACInteraction
from models.decoders.mtp import MTP
from models.decoders.multipath import Multipath
from models.decoders.covernet import CoverNet
from models.decoders.lvm import LVM
from models.decoders.future import Future
from models.decoders.query_tr import QueryTr

# Import metrics
from metrics.mtp_loss import MTPLoss
from metrics.min_ade import MinADEK
from metrics.min_fde import MinFDEK
from metrics.miss_rate import MissRateK
from metrics.covernet_loss import CoverNetLoss
from metrics.pi_bc import PiBehaviorCloning
from metrics.goal_pred_nll import GoalPredictionNLL
from metrics.LaplaceNLLLoss import LaplaceNLLLoss

from typing import List, Dict, Union


# Datasets
def initialize_dataset(dataset_type: str, args: List) -> TrajectoryDataset:
    """
    Helper function to initialize appropriate dataset by dataset type string
    """
    # TODO: Add more datasets as implemented
    dataset_classes = {'nuScenes_single_agent_raster': NuScenesRaster,
                       'nuScenes_single_agent_vector': NuScenesVector,
                       'nuScenes_single_agent_graphs': NuScenesGraphs,
                       }
    return dataset_classes[dataset_type](*args)


def get_specific_args(dataset_name: str, data_root: str, version: str = None) -> List:
    """
    Helper function to get dataset specific arguments.
    """
    # TODO: Add more datasets as implemented
    specific_args = []
    if dataset_name == 'nuScenes':
        ns = NuScenes(version, dataroot=data_root)
        pred_helper = PredictHelper(ns)
        specific_args.append(pred_helper)

    return specific_args


# Models
def initialize_prediction_model(encoder_type: str, aggregator_type: str, decoder_type: str,
                                encoder_args: Dict, aggregator_args: Union[Dict, None], decoder_args: Dict):
    """
    Helper function to initialize appropriate encoder, aggegator and decoder models
    """
    encoder = initialize_encoder(encoder_type, encoder_args)
    aggregator = initialize_aggregator(aggregator_type, aggregator_args)
    decoder = initialize_decoder(decoder_type, decoder_args)
    model = PredictionModel(encoder, aggregator, decoder)

    return model


def initialize_encoder(encoder_type: str, encoder_args: Dict):
    """
    Initialize appropriate encoder by type.
    """
    # TODO: Update as we add more encoder types
    encoder_mapping = {
        'raster_encoder': RasterEncoder,
        'polyline_subgraphs': PolylineSubgraphs,
        'pgp_encoder': PGPEncoder,
        'st_encoder': STEncoder,
        'pgp_mod_encoder': PGPModEncoder
    }

    return encoder_mapping[encoder_type](encoder_args)


def initialize_aggregator(aggregator_type: str, aggregator_args: Union[Dict, None]):
    """
    Initialize appropriate aggregator by type.
    """
    # TODO: Update as we add more aggregator types
    aggregator_mapping = {
        'concat': Concat,
        'global_attention': GlobalAttention,
        'gc': GoalConditioned,
        'pgp': PGP,
        'pass_through': PassThrough,
        'ac_aggregator': ACInteraction
        
    }
    if aggregator_type == 'pass_through':
        return aggregator_mapping[aggregator_type]()

    if aggregator_args:
        return aggregator_mapping[aggregator_type](aggregator_args)
    else:
        return aggregator_mapping[aggregator_type]()


def initialize_decoder(decoder_type: str, decoder_args: Dict):
    """
    Initialize appropriate decoder by type.
    """
    # TODO: Update as we add more decoder types
    decoder_mapping = {
        'mtp': MTP,
        'multipath': Multipath,
        'covernet': CoverNet,
        'lvm': LVM,
        'future': Future,
        'query': QueryTr
    }

    return decoder_mapping[decoder_type](decoder_args)


# Metrics
def initialize_metric(metric_type: str, metric_args: Dict = None):
    """
    Initialize appropriate metric by type.
    """
    # TODO: Update as we add more metrics
    metric_mapping = {
        'mtp_loss': MTPLoss,
        'covernet_loss': CoverNetLoss,
        'min_ade_k': MinADEK,
        'min_fde_k': MinFDEK,
        'miss_rate_k': MissRateK,
        'pi_bc': PiBehaviorCloning,
        'goal_pred_nll': GoalPredictionNLL,
        'LaplaceLoss': LaplaceNLLLoss
    }

    if metric_args is not None:
        return metric_mapping[metric_type](metric_args)
    else:
        return metric_mapping[metric_type]()

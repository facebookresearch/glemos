# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from enum import Enum, unique


@unique
class GraphDomain(Enum):
    # Domains from network repository
    BIO = "Biological Networks"
    BRAIN = "Brain Networks"
    CHEMICAL = "Chemical"
    CITATION = "Citation"
    COLLABORATION = "Collaboration"
    ECOLOGY = "Ecology"
    ECONOMIC = "Economic Networks"
    EMAIL = "Email"
    FACEBOOK = "Facebook Networks"
    INFRA = "Infrastructure"
    INTERACTION = "Interaction"
    POWER = "Power Networks"
    PROTEIN = "Protein"
    PROXIMITY = "Proximity"
    ROAD = "Road"
    RECOMMENDATION = "Recommendation"
    RETWEET = "Retweet"
    SC = "Scientific Computing"
    SOCIAL = "Social Networks"
    TECHNOLOGY = "Technology"
    TEMPREACH = "Temporal Reachability"
    WEB = "Web Graphs"
    # SYNTHETIC = "Synthetic"
    SYNTHETIC_BA = "Synthetic-BA"
    SYNTHETIC_KPGM = "Synthetic-KPGM"
    SYNTHETIC_CL = "Synthetic-CL"
    SYNTHETIC_ER = "Synthetic-ER"
    SYNTHETIC_SBM = "Synthetic-SBM"
    SYNTHETIC_RP = "Synthetic-RandPart"
    SYNTHETIC_OTHERS = "Synthetic-Others"
    PHONE = "Phone Call Networks"
    # Additional domains added for pyg graphs
    ACADEMIC = "Academic"
    COAUTHOR = "Coauthor"
    COPURCHASE = "Co-Purchase"
    COMPUTERVISION = "Computer Vision"
    FRIENDSHIP = "Friendship"  # friendship/followership
    FLIGHT = "Flight"
    KNOWLEDGEBASE = "Knowledgebase"
    WIKIPEDIA = "Wikipedia"
    MISC = "Misc"

    def __str__(self):
        return f"{self.value}"

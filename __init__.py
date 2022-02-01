import ROOT
from . import theoryCorrections,muonCorr,qcdUncByHelicity
ROOT.gInterpreter.Declare('#include "helicityWeightsToTensor.h"')

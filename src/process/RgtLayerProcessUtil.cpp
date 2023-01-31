#include "RgtLayerProcessUtil.h"

#include <limits>

double getNewRgtValueFromReference(long y, long z, long traceIndex, int rgtOriVal, float tdeb, float pasech, long dimy,
                const std::vector<ReferenceDuo>& referenceLayersVec, const std::vector<int>& refValues) {
	/*std::size_t index_ref = 0;
	while(index_ref<referenceLayersVec.size() && traceIndex>(referenceLayersVec[index_ref].iso[z*dimy+y]-tdeb)/pasech) {
		index_ref++;
	}
	double result;
	if(index_ref==0) {
		double v_Bottom_ref = refValues[0];
		if (referenceLayersVec[index_ref].rgt[z*dimy+y]!=0) {
			result = rgtOriVal * v_Bottom_ref/referenceLayersVec[index_ref].rgt[z*dimy+y]  ;
		} else {
			result = 0;
		}
	} else if (index_ref==referenceLayersVec.size()) {
		double v_Top_ref = refValues[refValues.size()-1];
		double v_Top = referenceLayersVec[refValues.size()-1].rgt[z*dimy+y];
		if (std::numeric_limits<short>::max() != v_Top) {
			result = v_Top_ref  + (rgtOriVal - v_Top) *(std::numeric_limits<short>::max() - v_Top_ref)/(std::numeric_limits<short>::max() - v_Top ) ;
		} else {
			result = v_Top_ref;
		}
	} else {
		double v_Top_ref = refValues[index_ref-1];
		double v_Top = referenceLayersVec[index_ref-1].rgt[z*dimy+y];
		double v_Bottom_ref = refValues[index_ref];
		double v_Bottom = referenceLayersVec[index_ref].rgt[z*dimy+y];
		if (v_Bottom!=v_Top) {
			result = v_Top_ref  + (rgtOriVal - v_Top) *(v_Bottom_ref - v_Top_ref)/(v_Bottom - v_Top) ;
		} else {
			result = v_Top_ref;
		}
	}
	return result;
	*/
	return rgtOriVal;
}

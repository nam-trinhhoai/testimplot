#ifndef SRC_SLICER_DATA_FIXEDLAYERS_FIXEDLAYERSFROMDATASETANDCUBE_Hpp
#define SRC_SLICER_DATA_FIXEDLAYERS_FIXEDLAYERSFROMDATASETANDCUBE_Hpp

template<typename InputType>
bool FixedLayersFromDatasetAndCube::checkValidity(const QByteArray& vect, std::size_t expectedSize) {
	return vect.size()>0 && vect.size()==expectedSize*sizeof(InputType) && vect.data()!=nullptr;
}

#endif

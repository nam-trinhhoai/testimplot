

template<typename InputType>
bool FixedRGBLayersFromDatasetAndCube::checkValidity(const QByteArray& vect, std::size_t expectedSize) {
	return vect.size()>0 && vect.size()==expectedSize*sizeof(InputType) && vect.data()!=nullptr;
}

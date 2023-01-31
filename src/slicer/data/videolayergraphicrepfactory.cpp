#include "videolayergraphicrepfactory.h"
#include "abstractinnerview.h"
#include "videolayer.h"
#include "videolayerrep.h"

VideoLayerGraphicRepFactory::VideoLayerGraphicRepFactory(
		VideoLayer *data) :
		IGraphicRepFactory(data) {
	m_data = data;
}

VideoLayerGraphicRepFactory::~VideoLayerGraphicRepFactory() {

}

AbstractGraphicRep* VideoLayerGraphicRepFactory::rep(ViewType type,
		AbstractInnerView *parent) {
	if (type == ViewType::BasemapView || type == ViewType::StackBasemapView) {
		VideoLayerRep* rep = new VideoLayerRep(m_data, parent);
		return rep;
	}
	return nullptr;
}


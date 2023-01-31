#include "ijkhorizongraphicrepfactory.h"
#include "abstractinnerview.h"
#include "ijkhorizon.h"
#include "cudaimagepaletteholder.h"
#include "affine2dtransformation.h"
#include "affinetransformation.h"


IJKHorizonGraphicRepFactory::IJKHorizonGraphicRepFactory(
		IJKHorizon *data) :
		IGraphicRepFactory(data) {
	m_data = data;
}

IJKHorizonGraphicRepFactory::~IJKHorizonGraphicRepFactory() {

}

AbstractGraphicRep* IJKHorizonGraphicRepFactory::rep(ViewType type,
		AbstractInnerView *parent) {
	return nullptr;
}

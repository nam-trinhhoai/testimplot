#ifndef GenericMaterialInitializer_H
#define GenericMaterialInitializer_H

#include <QString>
#include <QMaterial>

class GenericMaterialInitializer
{
public:
	virtual void initMaterial(Qt3DRender::QMaterial *m_material, QString pathVertex)=0;

	virtual void hide()=0;

};

#endif

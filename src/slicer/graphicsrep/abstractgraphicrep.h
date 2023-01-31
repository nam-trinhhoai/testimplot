#ifndef AbstractGraphicRep_H
#define AbstractGraphicRep_H

#include <QObject>
#include <QList>
#include "viewutils.h"
class QWidget;
class QGraphicsItem;
class QMenu;
class IData;
class GraphicLayer;
class Graphic3DLayer;
class QGraphicsScene;
class QWindow;
class AbstractInnerView;

namespace Qt3DCore {
	class QEntity;
}
namespace Qt3DRender
{
	class QCamera;
}

class AbstractGraphicRep:public QObject {
	Q_OBJECT
	Q_PROPERTY(QString name READ name CONSTANT WRITE setName NOTIFY nameChanged)
public:	
	enum TypeRep { NotDefined = 0, Courbe = 1, Image = 2, Video = 3, Image3D = 4, ImageRgt = 5};
	AbstractGraphicRep(AbstractInnerView *parent=nullptr);

	virtual ~AbstractGraphicRep();
	virtual bool canBeDisplayed()const;

	virtual QWidget* propertyPanel()=0;
	virtual GraphicLayer * layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent){return nullptr;}
	virtual Graphic3DLayer * layer3D(QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera){return nullptr;}

	virtual void buildContextMenu(QMenu * menu){};

	virtual QString name() const;
	virtual void setName(const QString & name);

	virtual IData* data() const=0;
	virtual void deleteLayer() {};
	AbstractInnerView *view()const{return m_parent;}
	virtual TypeRep getTypeGraphicRep() { return NotDefined;}
signals:
	//Signal used to dynamically insert a child in a viewer. Complementary to the working set and the data mecanism. To be used when no "real" data support is needed
	void insertChildRep(AbstractGraphicRep * rep);
	void nameChanged();
protected:
	AbstractInnerView *m_parent;
	QString m_name;

};
Q_DECLARE_METATYPE(AbstractGraphicRep *)
#endif

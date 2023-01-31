#include "helicoitem.h"
#include <cmath>
#include <QGraphicsView>

HelicoItem::HelicoItem(CameraParametersController* ctrl, QGraphicsItem *parent) : QGraphicsObject(parent)
{
	m_lock=true;
	m_controler = ctrl;
	m_helicoVisible = false;
	m_helicoItem = new QGraphicsSvgItem(":/slicer/icons/carrehelico2.svg",this);

	m_lineItem = new QGraphicsLineItem( m_helicoItem);
	QPen pen(Qt::red);
	pen.setWidth(2);
	pen.setCosmetic(true);
	m_lineItem->setPen(pen);
	m_lineItem->setLine( m_helicoItem->boundingRect().width()/2,m_helicoItem->boundingRect().height()/2.0f,m_helicoItem->boundingRect().width()/2,-5.0f);


	m_targetItem = new QGraphicsEllipseItem(0,0,400,400,m_lineItem);
	m_targetItem->setPen(pen);

	//m_helicoItem->setAcceptHoverEvents(true);
	//setAcceptHoverEvents(true);

	//m_helicoItem->setFlag(QGraphicsItem::ItemIsMovable, true);

	//m_helicoItem->setAcceptDrops(true);
	//setAcceptDrops(true);

	connect(m_controler,&CameraParametersController::posChanged,this,&HelicoItem::positionFromCtrl);


	connect(m_controler,&CameraParametersController::helicoShowed,this,&HelicoItem::showHelico);

	connect(m_controler,&CameraParametersController::targetChanged,this,&HelicoItem::targetChanged);

	connect(m_controler,&CameraParametersController::distanceTargetChanged,this,&HelicoItem::distanceTargetChanged);

	connect(m_helicoItem, &QGraphicsSvgItem::xChanged,this,&HelicoItem::positionChanged);

	connect(m_helicoItem, &QGraphicsSvgItem::yChanged,this,&HelicoItem::positionChanged);

	//connect(m_helicoItem, &QGraphicsSvgItem::wheelEvent,this,&HelicoItem::rotateWheel);
	//connect(m_helicoItem, &QGraphicsSvgItem::hoverEnterEvent,this,&HelicoItem::test);

}

HelicoItem::~HelicoItem()
{

}

void HelicoItem::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
	m_controler->showLineVert(false);
}

void HelicoItem::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
	m_controler->showLineVert(true);
}

void HelicoItem::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{

	QPointF posi= event->pos()-QPointF(m_helicoItem->boundingRect().width()/2,m_helicoItem->boundingRect().height()/2);

	m_helicoItem->setPos(posi);
}

QRectF HelicoItem::boundingRect() const
{
	/*QRectF rectH = m_helicoItem->boundingRect();

	QRectF rectL = m_lineItem->mapToParent(m_lineItem->boundingRect()).boundingRect();
	QRectF rectT = rectH.united( rectL);*/
	return  m_helicoItem->mapToParent( m_helicoItem->boundingRect()).boundingRect();
}

void HelicoItem::positionFromCtrl(QVector3D pos)
{
	transformItemZoomScale(pos);
}

void HelicoItem::targetChanged()
{
	transformItemZoomScale(m_controler->position());

}


void HelicoItem::wheelEvent(QGraphicsSceneWheelEvent *event)
{
//	qDebug()<<"wheel Event "<<boundingRect();
	//if(m_mouseOnHelico)
	//{
		prepareGeometryChange();
		rotateWheel(event->delta()/1.0f);

		event->accept();
/*	}
	else
	{

	}*/

}/*
void HelicoItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
	m_mouseOnHelico=true;
}
void HelicoItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
	m_mouseOnHelico=false;
}*/

void HelicoItem::test( QGraphicsSceneHoverEvent *event)
{
	qDebug()<<"test";
}

void HelicoItem::distanceTargetChanged(float d)
{
	if( this->scene() == nullptr) return;
	prepareGeometryChange();
	double svgScale = computeScale(this->scene());
	QRectF rect = m_helicoItem->boundingRect();
	QRectF recttarget = m_targetItem->boundingRect();
	m_targetItem->setPos(svgScale *(rect.width()/2-recttarget.width()/2.0),
		svgScale *(rect.height()/2.0f-recttarget.height()/2.0f)-m_controler->distanceTarget());

	m_lineItem->setLine(svgScale * rect.width()/2, svgScale *rect.height()/2.0f,svgScale * rect.width()/2,svgScale *rect.height()/2.0f-m_controler->distanceTarget()  );

}

void HelicoItem::positionChanged()
{
	if(qFuzzyCompare(m_posX,m_helicoItem->x()) && qFuzzyCompare(m_posY,m_helicoItem->y())) return;





	m_lock = false;

	prepareGeometryChange();

	float altitude = m_controler->position().y();

	QVector3D pos( m_helicoItem->x()+ m_helicoItem->boundingRect().width()/2, altitude,m_helicoItem->y()+ m_helicoItem->boundingRect().height()/2);
	QVector3D lastpos(m_lastPosition.x()+ m_helicoItem->boundingRect().width()/2,altitude,m_lastPosition.y() + m_helicoItem->boundingRect().height()/2);

	m_lastPosition = QPointF(m_helicoItem->x(),m_helicoItem->y());

	m_posX = m_helicoItem->x();
	m_posY = m_helicoItem->y();



	QVector3D dirCtrl = m_controler->target() -m_controler->position();
	QVector3D dirCtrl2D (dirCtrl.x(),0.0f,dirCtrl.z());
	float dirY = dirCtrl.y();

	QVector3D  dxdy = (pos - lastpos).normalized();
	m_indexTab = (m_indexTab+1)%m_tabDirection.size();
	m_tabDirection[m_indexTab] = dxdy ;//(pos - lastpos).normalized();

	float length = dirCtrl.length();


	QVector3D  newDir = dirCtrl2D.normalized()+ moyenneTab()*0.04f;
	QVector3D moy =newDir.normalized() *dirCtrl2D.length();

	moy.setY(dirY);
	QVector3D target =  pos + (moy);



	m_controler->requestPosChanged(pos);
	m_controler->requestTargetChanged(target);

	QVector2D right(1.0f,0.0);
	QVector2D up(0.0f,1.0);
	QVector2D dir = QVector2D(target.x(),target.z()) - QVector2D(pos.x(),pos.z()); //QVector2D (controler->target().x()-controler->position().x() ,controler->target().z()-controler->position().z());


	float angleRight=180.0f / M_PI * std::acos( QVector2D::dotProduct(dir.normalized(),right));

	float coef = -1.0f;
	if(angleRight> 90.0f) coef = 1.0f;
	float angle =coef * 180.0f / M_PI * std::acos( QVector2D::dotProduct(dir.normalized(),up));



	m_helicoItem->setRotation(180.0f+ angle);

	m_lock= true;

}

void HelicoItem::refreshItemZoomScale()
{
	transformItemZoomScale(m_controler->position());
}
void HelicoItem::transformItemZoomScale(QVector3D position, QGraphicsScene* scene)
{
	if( scene == nullptr && this->scene() == nullptr) return;
	if(m_lock== false) return;

		QSignalBlocker blocker(m_helicoItem);




		prepareGeometryChange();

		m_indexTab =0;
		m_tabDirection.resize(m_sizeTab);
		for(int i=0;i<m_tabDirection.size();i++)
		{
			m_tabDirection[i] = m_controler->target()- m_controler->position();
			m_tabDirection[i].setY(0.0f);
		}

		double svgScale = computeScale(scene);

		QRectF rect = m_helicoItem->boundingRect();

			QVector2D right(1.0f,0.0);
			QVector2D up(0.0f,1.0);
			QVector2D dir =QVector2D (m_controler->target().x()-m_controler->position().x() ,m_controler->target().z()-m_controler->position().z());
			float angleRight=180.0f / M_PI * std::acos( QVector2D::dotProduct(dir.normalized(),right));

			float coef = -1.0f;
			if(angleRight> 90.0f) coef = 1.0f;
			float angle =coef * 180.0f / M_PI * std::acos( QVector2D::dotProduct(dir.normalized(),up));

			m_lastPosition = QPointF(position.x(),position.z());

			m_helicoItem->setTransformOriginPoint(QPointF(rect.width() *0.5f,rect.height() *0.5f));

			//item->setPos(QPointF(position.x() - rect.width()/2, position.z() - rect.height()/2));
			m_helicoItem->setPos(QPointF(position.x() - rect.width()/2, position.z() - rect.height()/2));


			float distanceTarget= m_controler->distanceTarget();

			m_helicoItem->setScale(svgScale);
			m_helicoItem->setRotation(180.0f+ angle);

			//qDebug()<<" rotation angle :"<<angle;

			m_lineItem->setLine(svgScale * rect.width()/2, svgScale *rect.height()/2.0f,svgScale * rect.width()/2,svgScale *rect.height()/2.0f-distanceTarget  );
			m_lineItem->setScale(1.0f/svgScale);


			QRectF recttarget = m_targetItem->boundingRect();
			m_targetItem->setPos(svgScale *(rect.width()/2-recttarget.width()/2.0),
					svgScale *(rect.height()/2.0f-recttarget.height()/2.0f)-distanceTarget);

			m_targetItem->setScale(svgScale);

			m_posX = position.x() - rect.width()/2;
			m_posY = position.z() - rect.height()/2;
}

double HelicoItem::computeScale(QGraphicsScene* scene)
{

	QGraphicsView* view;
	if(scene != nullptr)
	{
		view =scene->views()[0];
	}
	else
	{
		view = this->scene()->views()[0];
	}
	QScreen* screen  =view->screen();
	float inchSize =0.3f;
	QPoint topLeft(0, 0);
	QPoint topRight(screen->logicalDotsPerInchX()*inchSize, 0);
	QPoint bottomLeft(0, screen->logicalDotsPerInchY()*inchSize);



	QRectF rect = m_helicoItem->boundingRect();


	QSize viewSize = view->viewport()->size();

	//	QPoint topLeft(0, 0), topRight(viewSize.width(), 0), bottomLeft(0, viewSize.height());
	QPoint topLeftWidget = view->mapFromGlobal(topLeft);
	QPoint topRightWidget = view->mapFromGlobal(topRight);
	QPoint bottomLeftWidget = view->mapFromGlobal(bottomLeft);

	QPointF topLeft1 = view->mapToScene(topLeftWidget);
	QPointF topRight1 = view->mapToScene(topRightWidget);
	QPointF bottomLeft1 = view->mapToScene(bottomLeftWidget);

	QPointF dWidthPt = (topRight1 - topLeft1);
	double dWidth = std::sqrt(std::pow(dWidthPt.x(),2) + std::pow(dWidthPt.y(), 2));
	QPointF dHeightPt = (bottomLeft1 - topLeft1);
	double dHeight = std::sqrt(std::pow(dHeightPt.x(),2) + std::pow(dHeightPt.y(), 2));


	double svgScale = std::min(dWidth / rect.width(), dHeight / rect.height());
	return svgScale;
}


void HelicoItem::showHelico(bool b)
{
	m_helicoVisible = b;
	if(m_helicoItem != nullptr)
	{
		m_helicoItem->setVisible(m_helicoVisible);
		//m_lineItem->setVisible(m_helicoVisible);
		//m_targetItem->setVisible(m_helicoVisible);
	}
}

QVector3D HelicoItem::moyenneTab()
{
	QVector3D res = QVector3D(0.0f,0.0f,0.0f);
	for(int i=0;i<m_tabDirection.size();i++)
	{
		res += m_tabDirection[i];
	}
	return res.normalized();

}

QVariant HelicoItem::itemChange(GraphicsItemChange change, const QVariant & value)
{
	 if (change == ItemSceneChange)
	 {
		 QGraphicsScene* scene = qvariant_cast<QGraphicsScene*>(value);
		 if(scene != nullptr)
		 {
			 scene->views()[0]->installEventFilter(this);
			 transformItemZoomScale(m_controler->position(),scene);
		 }
		 else
		 {
			 this->scene()->views()[0]->removeEventFilter(this);
		 }
	 }
	 return QGraphicsObject::itemChange(change,value);
}
/*
bool HelicoItem::eventFilter(QObject* watched, QEvent* ev) {
	//qDebug()<<"eventFilter...."<<ev->type();
	//if(m_mouseOnHelico==false){
		if (ev->type() == QEvent::Wheel) {
			qDebug()<<"transformItemZoomScale ";
			transformItemZoomScale(m_controler->position());
			return true;
		}
//	}
	return false;
}*/

void HelicoItem::rotateWheel(float coef)
{
	//m_angleWheel += coef;
	if(m_helicoVisible)
	{
	QVector3D dirCtrl = m_controler->target() -m_controler->position();
	QVector3D dirRight = QVector3D::crossProduct(dirCtrl.normalized(), QVector3D(0,1,0));
	QVector3D target = m_controler->target()+ dirRight.normalized()*3.0*coef;
	m_controler->requestTargetChanged(target);

	}


	//m_helicoItem->setRotation(180.0f+ m_angleWheel);
}

void HelicoItem::paint(QPainter* painter,const QStyleOptionGraphicsItem* option, QWidget *widget)
{

}

/*
 * GraphicSceneEditor.h
 *
 *  Created on: Sep 6, 2021
 *      Author: l1046262
 */

#ifndef SRC_GENERICEDITOR_GRAPHEDITOR_GRAPHICSCENEEDITOR_H_
#define SRC_GENERICEDITOR_GRAPHEDITOR_GRAPHICSCENEEDITOR_H_

#include <QObject>
#include <QGraphicsScene>
#include <QGraphicsItem>
#include <QGraphicsObject>
#include <QGraphicsSceneWheelEvent>
#include <QVariant>
#include <QThread>
#include <QPointer>

#include "abstract2Dinnerview.h"
#include "undosystem.h"
#include "GraphicToolsWidget.h"


QT_BEGIN_NAMESPACE
class QGraphicsSceneMouseEvent;
class QMenu;
class QPointF;
class QGraphicsLineItem;
class QFont;
class QGraphicsTextItem;
class QColor;
class GraphicTool_GraphicLayer;
class GraphEditor_LineShape;
class PointCtrl;
class GraphEditor_CurveShape;
class GraphEditor_Path;
class GraphEditor_RegularBezierPath;
class GraphEditor_ListBezierPath;

class BaseMapSurface;

QT_END_NAMESPACE

class GraphicSceneEditor : public QGraphicsScene
{
	Q_OBJECT
public:
	explicit GraphicSceneEditor(Abstract2DInnerView* innerview, QObject *parent = nullptr);
	void rotate(signed short);
	Abstract2DInnerView* innerView()
	{
		return m_InnerView;
	}
	void save_state(QString culturalName);
	void restore_state(QString culturalName);
	void hideItemsLayer(int  slicevalue);
	void showItemsLayer(int  slicevalue);
	void setSliceValue(int);
	void backupUndostack();
	void addLayer(QString , QList<QGraphicsItem *>);
	QList<QGraphicsItem*> CloneSceneItem();
	void saveItem(QGraphicsItem*pItem);

	void cloneDirectrice(GraphEditor_Path *);
	void clearDirectrice(QString name);

	QGraphicsItem* findItem(QString id);

	void directriceDeleted(QString);

	void positionCurrentChanged(QString id, float u);


	void applyColor(QString nameNurbs,QColor col);
	void deleteMyItem(int slice, QGraphicsItem* item);

	GraphEditor_ListBezierPath* addNewCurve(QVector<PointCtrl> listepoints, bool isopen,QString name);
	GraphEditor_CurveShape* addNewCurve(QPolygonF poly, bool isopen);
	GraphEditor_ListBezierPath* addNewCurve(GraphEditor_ListBezierPath* path);
	void deleteItem(QGraphicsItem*);
	GraphEditor_Path* getSelectedBezier(){ return m_bezierSelected;}

	GraphEditor_Path* getSelectBezier(QString nom );
	GraphEditor_Path* getCurrentBezier(QString nom );

	BaseMapSurface* cloneAndKeepFromLayerSpectrum();
	void deleteInsideData();

	void refreshWidthOrtho(int,int);
	void moveFinish(GraphEditor_Path*);
	GraphEditor_Path* findPath();
	GraphEditor_RegularBezierPath* findBezierPath();

	void setColorCustom(QColor);
	QColor getColorCustom();
	void setGeneratriceColor(QColor color ,QString name);


signals:
	void synchroniseScene(QList<QGraphicsItem*>, UndoSystem);
	void orthogonalUpdated(QPolygonF);
//	void sendIndexChanged(int);

	void updateWidthRandom(int,int);

	//void movePosition(float);
	void deleteNurbs(QString name);


	void sendDirectriceOk();





protected:
	void mousePressEvent(QGraphicsSceneMouseEvent *mouseEvent) override;
	void mouseReleaseEvent(QGraphicsSceneMouseEvent *mouseEvent) override;
	void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *mouseEvent) override;
	void mouseMoveEvent(QGraphicsSceneMouseEvent *mouseEvent) override;
	void keyPressEvent(QKeyEvent *event);

	public slots:
	//void test(float);
	void GraphicToolNewAction(eGraphicAction, st_GraphicToolsSettings);
	void updateSelectedItemsPen(QPen pen, ePenProperties);
	void updateSelectedItemsBrush(QBrush, eBrushProperties);
	void updateSelectedTextColor(QColor);
	void updateSlice(int);
	void updateSelectedCurveSmooth(int);
	void updateSelectedTextFont(QFont);
	void saveItemFromOtherScene(QList<QGraphicsItem*>, UndoSystem );

	void deleteItem();
	void onPolygonChanged(QVector<QPointF> v,bool);
	void onPolygonChangedTangent(QVector<PointCtrl> listeCtrl,QVector<QPointF> v,bool);
	void onPolygonChangedTangent(GraphEditor_ListBezierPath* path);
	//void receiveIndexCurrent(int);

	bool createOrgthognalRandomView(QString name ="",QColor col=Qt::yellow);
	void nurbs3d(QString name="",QColor col=Qt::yellow);
	void supprimerNurbs3d(QString name);

	void createListBezierPath(QString name,QVector<PointCtrl> listepts,QColor col,bool open);
	void onDeleted(QString);

	private slots:
	void copyItem();
	void pasteItem();
	void cutItem();

	void undo();
	void redo();
	void groupItems();
	void ungroupItems();
	void bringToFront();
	void sendToBack();
	void displayItemInfo();
	void deselectWells();
	void CreateRandomView();

	void deleteOrthoItem(GraphEditor_LineShape* pOrthItem);
	void cloneAndKeep();

	void path3d();

	//void createNurbs();
	void onSelected(GraphEditor_Path* p);




	private:


	QString getUniqueName();
	void createActions();
	void createMenus();
	void saveItem();
	void SceneStateMirror();
	void SceneStateMirror(QGraphicsItem * item);
	QGraphicsItem* createGraphicsItem(eShape , QPen , QBrush , int, bool antialiasingEnabled=true);
	void deleteItems(QList<QGraphicsItem*> const& items);
	QList<QGraphicsItem*> cloneItems(const QList<QGraphicsItem*>& items);
	//void deleteInsideData();
	QRectF CreateRect(QPointF p1, QPointF p2);
	QVariant item_to_variant(QGraphicsItem* item) ;
	QGraphicsItem* item_from_variant(QVariant v);

	bool m_scenerectchanged;
	QMenu *itemMenu;
	Abstract2DInnerView *m_InnerView;
	QGraphicsItem *m_item;
	QPolygonF PointsVec;
	int m_EndPicking=1;
	eGraphicAction m_CurrentGraphicAction;

	QList<QGraphicsItem*> m_PasteBoard;
	std::map <int,QList<QGraphicsItem*>> m_GraphicItems;
	std::map <int, GraphicTool_GraphicLayer*> m_LayersMap;
	UndoSystem undoStack;

	QAction *display;
	QAction *displayInfo;
	QAction *exitAction;
	QAction *addAction;
	QAction *deleteAction;
	QAction *copyAction;
	QAction *pasteAction;
	QAction *cutAction;
	QAction *undoAction;
	QAction *redoAction;
	QAction *m_Orthogonal;

	QAction *path3dAction;
	QAction *nurbs3dAction;
	//QAction *createNurbsAction;

	QAction *cloneAndKeepAction;
	QAction *toFrontAction;
	QAction *sendBackAction;
	QAction *groupAction;
	QAction *ungroupAction;
	QAction *deselectWellsAction;
	GraphEditor_LineShape *m_orthoLine;
	int m_SliceValue;
	QThread thread;

	QColor m_colorCustom= Qt::yellow;
	GraphEditor_Path* m_bezierSelected = nullptr;

	//QPointer<GraphEditor_Path> test;

	float m_randomWidth = 3000.0f;

};

#endif /* SRC_GENERICEDITOR_GRAPHEDITOR_GRAPHICSCENEEDITOR_H_ */

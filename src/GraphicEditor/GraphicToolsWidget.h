/*
 * GraphicToolsDialog.h
 *
 *  Created on: Aug 5, 2021
 *      Author: l1046262
 */

#ifndef SRC_SLICER_DIALOG_GRAPHICTOOLSWIDGET_H_
#define SRC_SLICER_DIALOG_GRAPHICTOOLSWIDGET_H_

#include <QWidget>
#include <QDialog>
#include <QtGui>
#include <QVector>
#include <QFont>
#include <QComboBox>

#include "baseqglgraphicsview.h"

class QCheckBox;
class QComboBox;
class QLabel;
class QSpinBox;
class QPushButton;
class QWidgetAction;
class QAction;
class QColor;
class QToolButton;
class QComboBox;
class QSlider;
class GraphicSceneEditor;
class Abstract2DInnerView;

const QString GraphicToolsWidgetTitle = "Graphic Tools";

typedef enum {
	eShape_Line = 0,
	eShape_Rect = 1,
	eShape_RoundRect = 2,
	eShape_Circle = 3 ,
	eShape_Ellipse = 4 ,
	eShape_Polygon = 5,
	eShape_Polyline = 6,
	eShape_Bezier_Curve = 7,
	eShape_CubicBSpline = 8,
	eShape_RegularBezierPath,
	eShape_ListBezierPath,
	eShape_SubDivideBezier,
	eShape_SubdivideBSpline,
	eShape_RationalBezier,
	eShape_NURBS,
	eShape_FreeDraw,
	eShape_Triangle = 0xFF //absolete
} eShape;

#define ISRECTSHAPE(X) 	(X==eShape_Rect)||(X==eShape_RoundRect)
#define EllipseFlag 0x2000
#define ISCUREVESHAPE(X) (X==eShape_Bezier_Curve)||(X==eShape_CubicBSpline)||(X==eShape_ListBezierPath)

typedef enum {
	eGraphicAction_Default,
	eGraphicAction_Draw,
	eGraphicAction_Erase,
	eGraphicAction_Fill,
	eGraphicAction_Copy,
	eGraphicAction_Paste,
	eGraphicAction_Cut,
	eGraphicAction_Delete,
	eGraphicAction_BringFront,
	eGraphicAction_SendBack,
	eGraphicAction_Undo,
	eGraphicAction_Redo,
	eGraphicAction_Text = 23,
} eGraphicAction;

typedef struct {
	bool enabled = false;
	// Active InnerVew (last one active)
	QString viewName;
	GraphicSceneEditor *pActiveScene = nullptr;
	eGraphicAction action = eGraphicAction_Default;
	eShape shape = eShape_Line;
	int smooth = 0;
	QPen pen;
	QBrush brush;
	QColor textColor;
	QFont font;
	bool antialiasingEnabled = false;
} st_GraphicToolsSettings;

typedef enum {
	e_BrushStyle,
	e_BrushColor
} eBrushProperties;

typedef enum {
	e_PenStyle,
	e_PenWidth,
	e_PenColor,
	e_PenCap,
	e_PenJoinStyle
} ePenProperties;

class GraphicToolsWidget :public QDialog{
	Q_OBJECT
public:

	static void showPalette(QString);
	static void closePalette();
	static st_GraphicToolsSettings getPaletteSettings();
	static void setDefaultAction();
	static void setActiveInnerView(QString innverView, GraphicSceneEditor *scene);
	static void removeInnerView(Abstract2DInnerView* pInnerview);

	void removeInnerView2(Abstract2DInnerView* pInnerview);

	static GraphicToolsWidget* getInstance(){
		if (!m_pInstance)
			{
				m_pInstance = new GraphicToolsWidget();
			}
		return m_pInstance;
	}


protected:
void closeEvent(QCloseEvent *event);


signals:
	void graphicOptionsChanged(eGraphicAction, st_GraphicToolsSettings);
	void penSettingsChanged(QPen, ePenProperties);
	void brushSettingsChanged(QBrush, eBrushProperties);
	void textColorChanged(QColor);
	void setViewCurrent(Abstract2DInnerView*);
	void textFontChanged(QFont);

private slots:
	void updateColorSelected();
	void updateBrushSelected();
	void updateToolSelected();
	void updateStyleSelected();
	void updateWidthSelected();
	void selectColorForBrush(bool);
	void selectColorForPen(bool);
	void selectColorForText(bool);
	void PenCapChanged(QAction* action);
	void PenJoinChanged(QAction* action);
	void setAntialiasingOption();
	void updateSelectedView();
	void updateCurveSmoothValue(int);
	void selectFont();
	void changeColor(QColor);

private:
	GraphicToolsWidget();
	virtual ~GraphicToolsWidget();

	void setDefaultSettings();
	void updateScenesSelectedItemsPen();
	void updateScenesSelectedItemsBrush(eBrushProperties);

	void changeGraphicsViewsDragMode(QGraphicsView::DragMode);
	void deselectTools();
	/*template <typename T>
	void updateSelectedItem(QVector<T> in_ActionVec, int &old_item_index); */

	static GraphicToolsWidget* m_pInstance;
	QLabel *m_SelectedColor_Label;
	//QCheckBox *m_AntialiasingCheckBox;
	QVector<QAction*> m_ShapeQActionVec;
	QVector<QAction*> m_ColorQActionVec;
	QVector<QAction*> m_BrushQActionVec;
	QVector<QAction*> m_ToolsQActionVec;
	QVector<QToolButton*> m_PenWidthToolButtonVec;
	QVector<QToolButton*> m_PenStyleToolButtonVec;
	QToolButton *m_BrushcolorSelection;
	QToolButton *m_PencolorSelection;
	QToolButton *m_TextcolorSelection;
	QToolButton *m_FontSelection;
	st_GraphicToolsSettings m_GraphicToolsSettings;
	//QComboBox *m_ViewCombobox;
	QSlider *m_SmoothSlider;
	QMenu *penWidthMenu;
	QMenu *penStyleMenu;
	QMenu *brushMenu;
	QVector<GraphicSceneEditor *> m_InnverViewsScene;
};


#endif /* SRC_SLICER_DIALOG_GRAPHICTOOLSWIDGET_H_ */

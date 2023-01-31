/*
 * MuratCanvas2dFullView.h
 *
 *  Created on: 16 janv. 2018
 *      Author: j0483271
 */

#ifndef MURATAPP_SRC_VIEW_CANVAS2D_MURATCANVAS2DFULLVIEW_H_
#define MURATAPP_SRC_VIEW_CANVAS2D_MURATCANVAS2DFULLVIEW_H_

#include "MuratCanvas2dView.h"

class MuratCanvas2dFullView: public MuratCanvas2dView {
	Q_OBJECT
public:
	MuratCanvas2dFullView(QWidget* parent=0);
	virtual ~MuratCanvas2dFullView();

	void initScene();



    /*
     * Public method
     *
     * Move arrow to position
     */
    void simulate_gentle_move(QPointF pos);

    /*
     * Public method
     *
     * Move arrow to position
     */
    void simulate_gentle_y_move(qreal yPos);

    void simulate_gentle_y_zoom(double factor, qreal y_target_scene_pos);

    bool getDisplayCurves();
    void setDisplayCurves(bool);
    bool getDisplayCursors();
    void setDisplayCursors(bool);

    virtual MuratCanvas2dScene* getCanvas() override;

protected:
    /*
     * Protected Class Method override QGraphicsView method
     *
     * Called when mouse move
     * Catch mouse mosition and launch updatePaddingArea
     */
    virtual void mouseMoveEvent(QMouseEvent*);

    virtual void mousePressEvent(QMouseEvent*);

    int x_move_center;
    int y_move_center=0;

private:
	/*
     * Private method
     *
     * Getter of canvas should be use to access canvas
     */

	bool displayCurves;
	bool displayCursors;

signals:
    void paletteRequestedSignal(MuratCanvas2dFullView*);
    void requestMenu(QPoint pos, MuratCanvas2dFullView*);

};

#endif /* MURATAPP_SRC_VIEW_CANVAS2D_MURATCANVAS2DFULLVIEW_H_ */

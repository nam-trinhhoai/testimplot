/*
 * Canvas2DExtension.h
 *
 *  Created on: 28 mars 2018
 *      Author: l0380577
 */

#ifndef MURATAPP_SRC_VIEW_CANVAS2D_CANVAS2DEXTENSION_H_
#define MURATAPP_SRC_VIEW_CANVAS2D_CANVAS2DEXTENSION_H_

class QGraphicsSceneMouseEvent;
class QGraphicsSceneWheelEvent;
class QKeyEvent;



class MuratCanvas2dScene;

class Canvas2DExtension {
public:
	Canvas2DExtension(){};
	virtual ~Canvas2DExtension(){};

	/**
	 * Mouse binding
	 */
	virtual void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *mouseEvent){};
	virtual void mouseMoveEvent(QGraphicsSceneMouseEvent *mouseEvent){};
	virtual void mousePressEvent(QGraphicsSceneMouseEvent *mouseEvent){};
	virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent *mouseEvent){};
	virtual void wheelEvent(QGraphicsSceneWheelEvent *wheelEvent){};

	/**
	 * Keyboard binding
	 */
	virtual void keyPressEvent(QKeyEvent *keyEvent){};
	virtual void keyReleaseEvent(QKeyEvent *keyEvent){};

	/**
	 * Initialize with the canvas.
	 * Add items, ...
	 */
	virtual void initCanvas(MuratCanvas2dScene* canvas){};
	virtual void releaseCanvas(MuratCanvas2dScene* canvas){};

	/**
	 * If an extension is disabled no event comes from the graphics scene
	 * Those methods remain virtual in case of some extension need to be aware that it has been deactivated
	 */
	virtual void setEnabled(bool enabled){
		_isEnabled=enabled;
	}
	virtual bool isEnabled(){
		return _isEnabled;
	}

protected:
	bool _isEnabled = true;

};


#endif /* MURATAPP_SRC_VIEW_CANVAS2D_CANVAS2DEXTENSION_H_ */

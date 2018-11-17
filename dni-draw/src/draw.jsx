import React, { Component } from 'react';
import CanvasDraw from 'react-canvas-draw';

export default class extends Component {
  constructor(props) {
    super(props);

    this.draw = React.createRef();
  }

  render() {
    return <div className="draw">
      <CanvasDraw
        lazyRadius={0}
        hideGrid={true}
        brushColor="#000"
        ref={this.draw}
      />
      <button
        onClick={() => this.draw.current.clear()}
      >Clear</button>
      <button
        onClick={() => console.log(this.draw.current.getSaveData())}
      >Log</button>
    </div>;
  }
}
